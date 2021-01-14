from hyper_resilient_experiments.alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
import spaceray
from ray import tune

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def objective(config):
    acc, model = pytorch_alexnet.cifar100_pt_objective(config)
    search_results = {'average_res': acc}
    tune.report(**search_results)
    return search_results

if __name__ == "__main__":
    args = Namespace(json='../standard.json', trials=10, out='pytorch_benchmark.csv')
    spaceray.run_experiment(args, objective, ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/raylogs", cpu=8,
                                start_space=0, mode="max")