"""Code to run a single batch of spaces on a GPU node on Theta"""
from argparse import ArgumentParser
import pickle
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
from ray import tune
from bi_tune import multi_train, model_attack


def run_batch(s):
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/hyperres_pickled_args", "rb")
    args = pickle.load(f)
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/hyperres_pickled_spaces", "rb")
    hyperspaces = pickle.load(f)
    for i in s:
        current_space = hyperspaces[i]
        optimizer = Optimizer(current_space)
        if args.model == "segmentation_cityscapes" or args.model == "segmentation_gis":
            search_algo = SkOptSearch(optimizer, ['learning_rate', 'epochs', 'batch_size', 'adam_epsilon'],
                                      metric='average_res', mode='max')
        else:
            search_algo = SkOptSearch(optimizer, ['learning_rate', 'dropout', 'epochs', 'batch_size', 'adam_epsilon'],
                                      metric='average_res', mode='max')
        analysis = tune.run(multi_train, search_alg=search_algo, num_samples=int(args.trials),
                            resources_per_trial={'cpu': 25, 'gpu': 1},
                            local_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/ray_results")
        df = analysis.results_df
        df_name = "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/hyper_resilient_results/" + args.out + "/"
        df_name += "space_"
        df_name += str(i)
        df_name += ".csv"
        df.to_csv(df_name)
        print("Finished space " + args.space)
    print("Finished all spaces. Files writtten to /lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/hyper_resilient_results/"
          + args.out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n")
    args = parser.parse_args()
    spaces = []
    for i in args.n:
        if i.isdigit():
            spaces.append(int(i))
    run_batch(spaces)
