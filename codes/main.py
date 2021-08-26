import nni
from collections import defaultdict
from main_function import main
import copy
import json
import numpy as np
import argparse

if __name__ == '__main__':
    data_series = "UDAGCN"   #UDAGCN  ACM_1 
    metric = "macro"  # macro micro 
    params = {"data_series" : data_series, "metric": metric}
        
    seeds = [1, 3, 5, 7, 9] # 1, 3, 5, 7, 9
    
    dataset_lists = {
        "UDAGCN": ["DBLP", "ACM"],
        "ACM_1": ["acm_a_1", "acm_b_1"],
    }
    dataset_list = dataset_lists[data_series] 
    score_record = defaultdict(list)
    result = 0.0
    params["dataset_list"] = dataset_list
    for i in range(len(dataset_list)):
        for j in range(len(dataset_list)):
            if i == j:
                continue
            source_dataset = dataset_list[i]
            target_dataset = dataset_list[j]

            params["source_dataset"] = source_dataset
            params["target_dataset"] = target_dataset

            for random_seed in seeds:
                params["random_seed"] = random_seed
                args, score_result, logger = main(params)   
                
                score_record[source_dataset + "_" + target_dataset].append(score_result)

    score_record = dict(score_record)
    avg_mean, avg_std = {"total":0}, {"total":0}

    for i in range(len(dataset_list)):
        for j in range(len(dataset_list)):
            if i == j:
                continue
            source_dataset = dataset_list[i]
            target_dataset = dataset_list[j]
            
            score = score_record[source_dataset + "_" + target_dataset]
            
            mean = np.mean(score) 
            std =  np.std(score) 
            
            avg_mean[source_dataset + "_" + target_dataset] = mean
            avg_std[source_dataset + "_" + target_dataset] = std
            avg_mean["total"] += mean
            avg_std["total"] += std

            print("{}: {}_{}:\tscore\t{:.4f}\tstd\t{:.4f}".format(params['metric'], source_dataset, target_dataset, mean, std))
    
    avg_mean["total"] /= (len(dataset_list) * (len(dataset_list) - 1))
    avg_std["total"] /= (len(dataset_list) * (len(dataset_list) - 1))
    
