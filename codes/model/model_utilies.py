import logging
import torch
from datetime import datetime
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from sklearn.metrics import f1_score
import numpy as np

def init_layer(args, params, conf_params):
    layer_unit_count_list = [args.num_features] 
    
    if args.num_layers == 2:
        layer_unit_count_list.extend([100])
    elif args.num_layers == 3:
        layer_unit_count_list.extend([256, 128])
    elif args.num_layers == 4:
        layer_unit_count_list.extend([256, 128, 64])
    elif args.num_layers == 5:
        layer_unit_count_list.extend([32, 32, 32, 32])
    elif args.num_layers == 7:
        layer_unit_count_list.extend([256, 128, 64, 32, 32,16])
    elif args.num_layers == 9: 
        layer_unit_count_list.extend([256, 128, 64, 64, 32, 32, 16, 16]) 
    layer_unit_count_list.append(args.num_label)
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vars(args)["device"] = device

    vars(args)["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    vars(args)["layer_unit_count_list"] = layer_unit_count_list
    
def init_log(args):
    with open("./record/" + args.model_name+f"/{args.random_seed}.log", 'w') as f:
        f.truncate()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("./record/" + args.model_name+f"/{args.random_seed}.log")
    fh.setLevel(logging.INFO)
    # ch = logging.StreamHandler(sys.stdout)
    # ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    logger.addHandler(fh)
    # logger.addHandler(ch)
    logger.info("logger name:%s", args.model_name + ".log")
    vars(args)["logger"] = logger
    return logger

def save_model(args, prefix, model):
    torch.save({'model_state_dict': model.state_dict()}, f"record/{args.model_name}/{prefix}_{args.random_seed}.pkl")

def load_model(args, prefix, model):
    state_dict = torch.load(f"record/{args.model_name}/{prefix}_{args.random_seed}.pkl", map_location=args.device)["model_state_dict"]
    model.load_state_dict(state_dict)
    return model

def Entropy(input):
    batch_size, num_feature = input.size()
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)

    return entropy 

def predict(output):
    labels = output.argmax(dim=1)
    
    return labels


def evaluate(output, labels, metric):
    preds = predict(output)
    corrects = preds.eq(labels)
    labels = labels.cpu().numpy()
    num_labels = np.max(labels) + 1
    preds = torch.argmax(output, dim = 1).cpu().numpy()
    macro_score = f1_score(labels, preds, average='macro')
    micro_score = f1_score(labels, preds, average='micro')

        
    if metric == "micro":
        score = micro_score
    elif metric == "macro":
        score  = macro_score
    else:
        print("wrong!")
        exit()

    return score


def cos_distance(input1, input2):
    norm1 = torch.norm(input1, dim = -1)
    norm2 = torch.norm(input2, dim = -1)
        
    norm1 = torch.unsqueeze(norm1, dim = 1)
    norm2 = torch.unsqueeze(norm2, dim = 0)

    cos_matrix = torch.matmul(input1, input2.t())
        
    cos_matrix /= norm1
    cos_matrix /= norm2

    return cos_matrix

def test(model, args, data, criterion, mode = 'valid'):
    outputs = model(data)

    if mode == 'valid':
        outputs = outputs[data.val_mask]
        labels = data.y[data.val_mask]
    else:
        labels = data.y

    loss = criterion(outputs, labels)
    acc = evaluate(outputs, labels, args.metric)

    return loss, acc

def generate_one_hot_label(labels):
    num_labels = torch.max(labels).item() + 1
    num_nodes = labels.shape[0]
    label_onehot = torch.zeros((num_nodes, num_labels)).cuda()
    label_onehot = F.one_hot(labels, num_labels).float().squeeze(1) 

    return label_onehot

def generate_normalized_adjs(adj, D_isqrt):
    DAD = D_isqrt.view(-1,1)*adj*D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1)*adj
    AD = adj*D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD

def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    return adj, deg_inv_sqrt