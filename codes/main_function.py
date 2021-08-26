from __future__ import print_function
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train_procedure import train_procedure
from model.model_utilies import init_layer, init_log
import utilies
import os
from datetime import datetime
from torch.nn.utils import weight_norm as weight_normalization
import time
from model.model import Model
from model.Data import DomainData
import time
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected
import pandas as pd
import pickle
from torch_sparse import SparseTensor

def main(params):
    parser = argparse.ArgumentParser(description='PyTorch Cora Example')
    parser.add_argument('--is_reading_conf', type=int, default=0, help='whether read the config file')
    parser.add_argument('--conf_name', type=str, default="test")
    parser.add_argument('--source_dataset', type=str, default='ACM',help='name of dataset')
    parser.add_argument('--target_dataset', type=str, default='DBLP',help='name of dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2,help='ratio of validation set')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers of the network')
    parser.add_argument('--source_lr', type=float, default=0.01,
                        help='input learning rate for training')
    parser.add_argument('--target_lr', type=float, default=0.01,
                        help='input learning rate for training')
    parser.add_argument('--source_epochs', type=int, default=101,
                        help='input training epochs for training (default: 101)')
    parser.add_argument('--target_epochs', type=int, default=101,
                        help='epochs for target_epochs  (default: 101)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='input random seed for training (default: 1)')
    
    # common settings
    parser.add_argument("--logname", type=str, default="info")
    parser.add_argument("--is_baseline", type=str, default=False)

    
    # whether train source
    parser.add_argument("--is_source_train", type=int, default=1)
    parser.add_argument("--source_model_path", type=str, default=" ")
    parser.add_argument("--model_name", type=str, default="hyper_select")


    # shot specific
    parser.add_argument('--struct_lambda', type=float, default=1,help='Structure NCE loss')
    parser.add_argument('--neigh_lambda', type=float, default=1,help='Neighborhood NCE loss')
    parser.add_argument('--gnn_model', type=str, default="GCN",help='Select different model')
    parser.add_argument('--head', type=int, default=1)  #2 4
    


    # set common config
    args = parser.parse_args()

    conf_params = {}
    if args.is_reading_conf:
        conf_params = utilies.load_json_file("./config/"+ args.conf_name + f".json")
    # params 是NNI和json文件传来的参数 NNI的优先级更高
    for key in conf_params:
        vars(args)[key] = conf_params[key]

    for key in params:
        vars(args)[key] = params[key]

    
    dataset_list = params["dataset_list"]
    # 补充两个数据集的方法
    num_nodes_dict = {"DBLP": 5578, "ACM": 7410, "acm_a_1": 1500, "acm_b_1": 1500}
    
    num_label_dict = {"DBLP": 6, "ACM": 6, "acm_a_1": 4, "acm_b_1": 4}
    
    num_feature_dict = {"DBLP": 7537, "ACM": 7537, "acm_a_1": 300, "acm_b_1": 300}

    single_dict = {"DBLP": True, "ACM": True, "acm_a_1":True, "acm_b_1": True}
    
    vars(args)['single'] = single_dict[args.source_dataset]
    vars(args)['num_label'] = num_label_dict[args.source_dataset]
    vars(args)['num_features'] = num_feature_dict[args.source_dataset]

    init_layer(args, params, conf_params)
    
    args.model_name = f"{args.model_name}_{args.gnn_model}_{args.num_layers}_{args.data_series}_{args.metric}_{args.head}_{args.struct_lambda}__{args.neigh_lambda}"
    args.domain_name = f"{args.source_dataset}_{args.target_dataset}"
    if not os.path.exists("./record/" + args.model_name):
        os.makedirs("./record/" + args.model_name)
    
    if not os.path.exists(f"./record/{args.model_name}/{args.domain_name}"):
        os.makedirs(f"./record/{args.model_name}/{args.domain_name}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set random seed
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    source_data = DomainData("../data/{}".format(args.source_dataset), name=args.source_dataset, valid_ratio = args.val_ratio)[0]
    target_data = DomainData("../data/{}".format(args.target_dataset), name=args.target_dataset)[0]

    


    if args.source_dataset in dataset_list:
        vars(args)['num_source_nodes'] = num_nodes_dict[args.source_dataset]
        source_data = DomainData("../data/{}".format(args.source_dataset), name=args.source_dataset, valid_ratio = args.val_ratio)
        source_data = source_data[0]
        print("source data: ", source_data)
        source_data = source_data.to(device)
    else:
        print(r'Dataset need to be defined!')
        sys.exit()
    
    if args.target_dataset in dataset_list:
        vars(args)['num_target_nodes'] = num_nodes_dict[args.target_dataset]
        target_data = DomainData("../data/{}".format(args.target_dataset), name=args.target_dataset)
        target_data = target_data[0]
        print("target data: ", target_data)
        target_data = target_data.to(device)
    else:
        print(r'Dataset need to be defined!')
        sys.exit()

    
    target_structure_data = target_data.clone()
    with open(f"../structure_data/{args.target_dataset}/4.txt", 'rb') as f:
        structure_link = torch.tensor(pickle.load(f))
    edge_idx = to_undirected(structure_link, args.num_target_nodes)
    target_data.edge_idx = edge_idx
    row, col = edge_idx
    structure_adj = SparseTensor(row=row, col=col, sparse_sizes=(args.num_target_nodes, args.num_target_nodes))
    
    logger = init_log(args)
    for key in vars(args):
        logger.info("{}:{}".format(key, vars(args)[key]))
    
    model = Model(args, logger)

    model.to(device)
    source_optimizer = optim.Adam(model.parameters(), lr=args.source_lr, weight_decay=5e-4)  # , momentum=0.9
    target_optimizer = optim.Adam(model.parameters(), lr=args.target_lr, weight_decay=5e-4)  # , momentum=0.9

    criterion = nn.CrossEntropyLoss()

    test_score = train_procedure(args, logger, model, source_optimizer, target_optimizer, criterion, source_data, target_data, target_structure_data, structure_adj)
    

    return args, test_score, logger

