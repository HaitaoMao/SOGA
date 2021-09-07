from collections import defaultdict
from main_function import main
import numpy as np

if __name__ == '__main__':
    source_domain = "acm_a_1"
    target_domain = "acm_b_1"
    metric = "macro"  # macro micro 
    params = {"source_dataset": source_domain, "target_dataset": target_domain, "metric": metric}
        
    seeds = [1, 3, 5, 7, 9] # 1, 3, 5, 7, 9
    
    score_list = []
    for random_seed in seeds:
        params["random_seed"] = random_seed
        args, score_result, logger = main(params)   

    print("{}: {}_{}:\tscore\t{:.4f}\tstd\t{:.4f}".format(params['metric'], params["source_dataset"], params["target_dataset"], np.mean(score_result), np.std(score_result)))
   