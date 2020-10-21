### Helper script to run experiments on top and bottom configurations and plot output layer weights for different models

from simple_mnist import mxnet_mnist, pt_mnist, tf_mnist
from alexnet_cifar import mxnet_alexnet, pytorch_alexnet, tensorflow_alexnet
from tqdm import tqdm
import statistics
import numpy as np
import pandas as pd
import sys

def get_best_config(results: pd.DataFrame):
    """ Given a results dataframe from the hyperspace tuning, get best and worst configs"""
    sorted_res = results.sort_values("average_res", ascending=False)
    sorted_res = sorted_res.reset_index(drop=True)
    top_config = {}
    bottom_config = {}
    last_index = len(results)-1
    for label in ['config.learning_rate', 'config.dropout', 'config.epochs', 'config.batch_size']:
        new_label = label.split('config.')[-1]
        top_config[new_label] = sorted_res[label][0]
        bottom_config[new_label] = sorted_res[label][last_index]
    return (top_config, bottom_config)

def pytorch_experiment(config, trials=50):
    pass

def tensorflow_experiment(config, trials=50):
    pass

def mxnet_experiment(config, trials=50):
    pass

def plot_curves(config, model_type, output="out.png", trials=50):
    """ Given a config and a model type, perform 50 trials and plot average output layer weights"""
    if model_type == "mnist":
        PT_MODEL = pt_mnist.mnist_pt_objective
        TF_MODEL = tf_mnist.mnist_tf_objective
        MX_MODEL = mxnet_mnist.mnist_mx_objective
    elif model_type == "alexnet_cifar100":
        PT_MODEL = pytorch_alexnet.cifar_pt_objective
        TF_MODEL = tensorflow_alexnet.cifar_tf_objective
        MX_MODEL = mxnet_alexnet.cifar_mxnet_objective
    else:
        print("Error: improper model type, must be either 'mnist' or 'alexnet_cifar100'. Please try again.")
        sys.exit()
    # run trials and collect output weights
