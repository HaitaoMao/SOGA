import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GINConv
from torch_geometric.nn.norm import LayerNorm 
import torch.nn.utils.weight_norm as weightNorm



class Extractor(nn.Module):
    def __init__(self, layer_unit_count_list, args):
        super(Extractor, self).__init__()
        
        self.layer_unit_count_list = layer_unit_count_list
        self.layer_count = len(self.layer_unit_count_list) - 1
        self.conv_layers, self.activate_layer = nn.ModuleList(), nn.ModuleList()

        for i in range(self.layer_count):
            if args.gnn_model == "GCN":
                self.conv_layers.append(GCNConv(self.layer_unit_count_list[i], self.layer_unit_count_list[i + 1]))
            elif args.gnn_model == "SAGE":
                self.conv_layers.append(SAGEConv(self.layer_unit_count_list[i], self.layer_unit_count_list[i + 1]))
            elif args.gnn_model == "GAT":
                self.conv_layers.append(GATConv(self.layer_unit_count_list[i], self.layer_unit_count_list[i + 1] // args.head, heads=args.head))


            self.activate_layer.append(
                nn.ReLU(),
            )        

    def forward(self, x, edge_idx, mask=None):
        batch_size, feature_dim = x.size()
        for i in range(self.layer_count):
            x = self.activate_layer[i](self.conv_layers[i](x, edge_idx))
            
        if mask:
            x = x[mask]
            
        return x


class Classifier(nn.Module):
    def __init__(self, layer_unit_count_list):
        super(Classifier, self).__init__()
        self.layer_unit_count_list = layer_unit_count_list
        self.fc = nn.Linear(self.layer_unit_count_list[0], self.layer_unit_count_list[1])
    
    def forward(self, x):
        batch_size, feature_dim = x.size()
        x = self.fc(x)

        return x



class CrossEntropy(nn.Module):
    def __init__(self, args, epsilon=0.1, reduction=True):
        super(CrossEntropy, self).__init__()
        self.num_classes = args.num_label
        self.epsilon = epsilon
        self.use_gpu = torch.cuda.is_available()
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        batch_size, feature_dim = outputs.size()
        log_probs = self.logsoftmax(outputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss

    
            

    