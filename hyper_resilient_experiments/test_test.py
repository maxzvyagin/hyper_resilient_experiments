import numpy as np
import argparse
import torch
import spaceray
from ray import tune

def objective(config):
    average_res = np.random.rand()
    search_results['average_res'] = average_res
    tune.report(**search_results)
    return search_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    # print(NUM_CLASSES)
    args = parser.parse_args()
    spaceray.run_experiment(args, objective, cpu=8, ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/raylogs")