""" Given a csv file of Ray Tune results, train models using the top and bottom configurations sorted by average_res,
save models to given directory """

import sys
sys.path.append("..")
import tensorflow as tf
import torch
from simple_mnist import pt_mnist, tf_mnist
from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
import argparse
import pandas as pd
import os

PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective

def train_models(i, o):
    try:
        os.mkdir(o)
    except:
        "NOTE: Error creating output directory "+o
    ray_results = pd.read_csv(i)
    sorted_ray_results = ray_results.sort_values('average_res', ascending=False)
    sorted_ray_results = sorted_ray_results.reset_index(drop=True)
    # top config
    print("Training top configuration...")
    top_config = {'learning_rate': float(sorted_ray_results['config.learning_rate'][0]),
                  'dropout': float(sorted_ray_results['config.dropout'][0]),
                  'epochs': int(sorted_ray_results['config.epochs'][0]),
                  'batch_size': int(sorted_ray_results['config.batch_size'][0])}
    top_pt_test_acc, pt_model = PT_MODEL(top_config)
    top_tf_test_acc, tf_model = TF_MODEL(top_config)
    torch.save(pt_model, o+"/top_pt_model")
    tf_model.save(o+"/top_tf_model")
    # bottom config
    print("Training bottom configuration...")
    i = len(sorted_ray_results)-1
    bottom_config = {'learning_rate': float(sorted_ray_results['config.learning_rate'][i]),
                     'dropout': float(sorted_ray_results['config.dropout'][i]),
                     'epochs': int(sorted_ray_results['config.epochs'][i]),
                     'batch_size': int(sorted_ray_results['config.batch_size'][i])}
    bottom_pt_test_acc, pt_model = PT_MODEL(bottom_config)
    bottom_tf_test_acc, tf_model = TF_MODEL(bottom_config)
    torch.save(pt_model, o + "/bottom_pt_model")
    tf_model.save(o + "/bottom_tf_model")
    # write out accuracies
    f = open(o+"/test_accuracies.txt", "w")
    f.write("Top PyTorch Accuracy: "+str(top_pt_test_acc)+"\n")
    f.write("Top TensorFlow Accuracy: "+str(top_tf_test_acc)+"\n")
    f.write("Bottom PyTorch Accuracy: "+str(bottom_pt_test_acc)+"\n")
    f.write("Bottom TensorFlow Accuracy: "+str(bottom_tf_test_acc)+"\n")
    f.close()

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
    train_models(args.input, args.output)