"""Code to run a single batch of spaces on a GPU node on Theta"""
from argparse import ArgumentParser
import pickle
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
from ray import tune
from bi_tune import multi_train, model_attack
from simple_mnist import pt_mnist, tf_mnist
from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from segmentation import pytorch_unet, tensorflow_unet
import sys

PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True
NODES = 4

def process_args(args):
    """Setting global variables using arguments"""
    global PT_MODEL, TF_MODEL, NUM_CLASSES, NO_FOOL, NODES, MNIST, TRIALS
    if not args.model:
        print("NOTE: Defaulting to MNIST model training...")
        args.model = "mnist"
    else:
        if args.model == "alexnet_cifar100":
            PT_MODEL = pytorch_alexnet.cifar_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar_tf_objective
            NUM_CLASSES = 100
        ## definition of gans as the model type
        elif args.model == "gan":
            print("Error: GAN not implemented.")
            sys.exit()
        elif args.model == "segmentation_cityscapes":
            PT_MODEL = pytorch_unet.cityscapes_pt_objective
            TF_MODEL = tensorflow_unet.cityscapes_tf_objective
            NUM_CLASSES = 30
        elif args.model == "segmentation_gis":
            PT_MODEL = pytorch_unet.gis_pt_pbjective
            TF_MODEL = tensorflow_unet.gis_tf_objective
            NUM_CLASSES = 1
        elif args.model == "mnist_nofool":
            NO_FOOL = True
        elif args.model == "cifar_nofool":
            NO_FOOL = True
            PT_MODEL = pytorch_alexnet.cifar_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar_tf_objective
            NUM_CLASSES = 100
        elif args.model == "alexnet_cifar10":
            PT_MODEL = pytorch_alexnet.cifar10_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar10_tf_objective
            NUM_CLASSES = 10
            MNIST = False
        elif args.model == "cifar10_nofool":
            NO_FOOL = True
            PT_MODEL = pytorch_alexnet.cifar10_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar10_tf_objective
            NUM_CLASSES = 10
        else:
            print("\n ERROR: Unknown model type. Please try again. "
                  "Must be one of: mnist, alexnet_cifar100, segmentation_cityscapes, or segmentation_gis.\n")
            sys.exit()
    if not args.trials:
        print("NOTE: Defaulting to 25 trials per scikit opt space...")
    else:
        TRIALS = int(args.trials)
    if not args.nodes:
        print("NOTE: Defaulting to 4 nodes per space...")
    else:
        NODES = int(args.nodes)


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
