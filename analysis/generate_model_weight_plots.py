""" Given a csv file of Ray Tune results, train models using the top and bottom configurations sorted by average_res,
save models to given directory """

import tensorflow as tf
import torch as
from ..simple_mnist import pt_mnist, tf_mnist
from ..alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
import argparse
import pandas as pd

PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective

def train_models(i, o):
    ray_results = pd.read_csv(i)
    sorted_ray_results = ray_results.sort_values('average_res')
    sorted_ray_results = sorted_ray_results.reset_index(drop=True)
    top_config = {}
    top_config['learning_rate'] = float(sorted_ray_results[0]['config.learning_rate'])


    # train pytorch model, then train tens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="Input CSV from Ray Tune results.")
    parser.add_argument('-o', '--output', required=True, help="Output directory for pickled models.")
    parser.add_argument('-m', '--model', required=True, help="Specify what type of model is being trained.")
    args = parser.parse_args()
    if args.model == "mnist":
        pass
    elif args.model == "cifar10":
        PT_MODEL = pytorch_alexnet.cifar10_pt_objective
        TF_MODEL = tensorflow_alexnet.cifar10_tf_objective
    elif args.model == "cifar100":
        PT_MODEL = pytorch_alexnet.cifar_pt_objective
        TF_MODEL = tensorflow_alexnet.cifar_tf_objective
