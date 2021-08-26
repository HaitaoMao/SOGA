from .basic_blocks import Extractor, Classifier, CrossEntropyLabelSmooth
from .model_utilies import Entropy, evaluate
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.NCE_utilies import RandomWalker, Negative_Sampler
from torch_geometric.utils.convert import to_networkx
import time

class Model(nn.Module):
    def __init__(self, args, logger):
        super(Model, self).__init__()
        self.layer_unit_count_list = args.layer_unit_count_list
        self.layer_count = len(self.layer_unit_count_list)
        self.logger = logger
        self.args = args
        
        
        self.Extractor = Extractor(self.layer_unit_count_list[:-1], args)
        self.Classifier = Classifier(self.layer_unit_count_list[-2:])
        
        self.models = nn.ModuleList([self.Extractor, self.Classifier])
        
        self.num_negative_samples = 5
        self.num_positive_samples = 2

        self.CrossEntropyLabelSmooth = CrossEntropyLabelSmooth(args)


    def init_target(self, graph_struct, graph_neigh):
        self.target_G_struct = to_networkx(graph_struct)
        self.target_G_neigh = to_networkx(graph_neigh)
        
        self.Positive_Sampler = RandomWalker(self.target_G_struct, p=0.25, q=2, use_rejection_sampling=1)
        self.Negative_Sampler = Negative_Sampler(self.target_G_struct)
        self.center_nodes_struct, self.positive_samples_struct = self.generate_positive_samples()
        self.negative_samples_struct = self.generate_negative_samples()

        self.Positive_Sampler = RandomWalker(self.target_G_neigh, p=0.25, q=2, use_rejection_sampling=1)
        self.Negative_Sampler = Negative_Sampler(self.target_G_struct)
        self.center_nodes_neigh, self.positive_samples_neigh = self.generate_positive_samples()
        self.negative_samples_neigh = self.generate_negative_samples()


    def generate_positive_samples(self):
        self.Positive_Sampler.preprocess_transition_probs()        
        self.positive_samples = self.Positive_Sampler.simulate_walks(num_walks=1, walk_length=self.num_positive_samples, workers=1, verbose=1)
        for i in range(len(self.positive_samples)):
            if len(self.positive_samples[i]) != 2:
                self.positive_samples[i].append(self.positive_samples[i][0])

        samples = torch.tensor(self.positive_samples).cuda()

        center_nodes = torch.unsqueeze(samples[:, 0], dim = -1)
        positive_samples = torch.unsqueeze(samples[:, 1], dim = -1)

        return center_nodes, positive_samples

    def generate_negative_samples(self):
        negative_samples = torch.tensor([self.Negative_Sampler.sample() for _ in range(self.num_negative_samples * self.args.num_target_nodes)]).view([self.args.num_target_nodes, self.num_negative_samples]).cuda()

        return negative_samples


    def ent(self, softmax_output):
        entropy_loss = torch.mean(Entropy(softmax_output))

        return entropy_loss

    def div(self, softmax_output):
        mean_softmax_output = softmax_output.mean(dim = 0)
        diversity_loss = torch.sum(-mean_softmax_output * torch.log(mean_softmax_output + 1e-8))
        
        return diversity_loss
    

    def train_source(self, source_data, optimizer, criterion, epoch):
        self.enable_source()
        
        outputs = self.forward(source_data)[source_data.train_mask]
        
        labels = source_data.y[source_data.train_mask]
        loss = self.CrossEntropyLabelSmooth(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), evaluate(outputs, labels, self.args.single, self.args.metric)

    
    
    def train_target(self, target_data, target_structure_data, structure_adj, criterion, optimizer, epoch):
        self.enable_target()

        outputs = self.forward(target_data)
        probs = F.softmax(outputs, dim = -1)
        if epoch == 0:
            source_target_acc = evaluate(outputs, target_data.y, self.args.single, self.args.metric)
            self.logger.info(f"accuracy of source model in target domain: {source_target_acc}")
        NCE_loss_struct = self.NCE_loss(probs, self.center_nodes_struct, self.positive_samples_struct, self.negative_samples_struct)
        NCE_loss_neigh = self.NCE_loss(probs, self.center_nodes_neigh, self.positive_samples_neigh, self.negative_samples_neigh)

        IM_loss = self.ent(probs) - self.div(probs)
        
        loss =  IM_loss + self.args.struct_lambda * NCE_loss_struct  + self.args.neigh_lambda * NCE_loss_neigh 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return evaluate(outputs, target_data.y, self.args.single, self.args.metric)
    

    
    def NCE_loss(self, outputs, center_nodes, positive_samples, negative_samples):
        negative_embedding = F.embedding(negative_samples, outputs)
        positive_embedding = F.embedding(positive_samples, outputs)
        center_embedding = F.embedding(center_nodes, outputs)


        positive_embedding = positive_embedding.permute([0, 2, 1])
        positive_score =  torch.bmm(center_embedding, positive_embedding).squeeze()
        exp_positive_score = torch.exp(positive_score).squeeze()
        
        negative_embedding = negative_embedding.permute([0, 2, 1])
        negative_score = torch.bmm(center_embedding, negative_embedding).squeeze()
        exp_negative_score = torch.exp(negative_score).squeeze()
        
        exp_negative_score = torch.sum(exp_negative_score, dim = 1)
        
        
        loss = -torch.log(exp_positive_score / exp_negative_score) 
        loss = loss.mean()

        return loss


    def forward(self, data):
        x = self.Extractor(data.x, data.edge_index)
        x = self.Classifier(x)
        
        return x

    def disable(self):
        for model in self.models:
            model.eval()
    
    def enable_source(self):
        for model in self.models:
            model.train()
        
    def enable_target(self):
        self.Extractor.train()
        # self.Classifier.train()
        self.Classifier.train()
        
    