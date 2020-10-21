### General tuning script to combine all submodules ###
### Works with MXNet in addition to TensorFlow and PyTorch ###
### CURRENTLY IN PROGRESS, NOT FUNCTIONAL ###

from simple_mnist import mxnet_mnist, pt_mnist, tf_mnist
from alexnet_cifar import mxnet_alexnet, pytorch_alexnet, tensorflow_alexnet
import argparse
from hyperspace import create_hyperspace
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
from tqdm import tqdm
import statistics
import foolbox as fb
import mxnet as mx
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Default function definitions
PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
MX_MODEL = mxnet_mnist.mnist_mx_objective

NUM_CLASSES = 10

def model_attack(model, model_type, attack_type, config):
    if model_type == "pt":
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), num_classes=NUM_CLASSES)
    elif model_type == "tf":
        fmodel = fb.models.TensorFlowEagerModel(model, bounds=(0, 1), num_classes=NUM_CLASSES)
    else:
        gpus = mx.test_utils.list_gpus()
        ctx = [mx.gpu(0)] if gpus else [mx.cpu(0)]
        fmodel = fb.models.MXNetGluonModel(model, bounds=(0,1), num_classes=NUM_CLASSES, ctx=ctx)
    if NUM_CLASSES == 10:
        # train, test = keras.datasets.cifar100.load_data()
        # images, labels = test
        images, labels = fb.utils.samples(dataset='mnist', batchsize=config['batch_size'], data_format='channels_last',
                                               bounds=(0, 1))
        #images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=config['batch_size'])
    else:
        # train, test = keras.datasets.mnist.load_data()
        # images, labels = test
        images, labels = fb.utils.samples(dataset='cifar100', batchsize=config['batch_size'],
                                          data_format='channels_last', bounds=(0, 1))
        #images, labels = fb.utils.samples(fmodel, dataset='cifar100', batchsize=config['batch_size'])
    if attack_type == "uniform":
        attack = fb.attacks.AdditiveUniformNoiseAttack(model=fmodel)
    elif attack_type == "gaussian":
        attack = fb.attacks.AdditiveGaussianNoiseAttack(model=fmodel)
    elif attack_type == "saltandpepper":
        attack = fb.attacks.SaltAndPepperNoiseAttack(model=fmodel)
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    adv = attack(images, labels, epsilons=epsilons)
    print(type(adv))
    if model_type == "pt":
        robust_accuracy = 1 - success.cpu().numpy().astype(float).flatten().mean(axis=-1)
    elif model_type == "tf":
        robust_accuracy = 1 - success.numpy().astype(float).flatten().mean(axis=-1)
    else:
        robust_accuracy = 1 - success.numpy().astype(float).flatten().mean(axis=-1)
    return robust_accuracy


def multi_train(config):
    config = {'epochs':1, 'batch_size': 64, 'learning_rate':.001, 'dropout':.5}
    print("PyTorch\n\n")
    pt_test_acc, pt_model = PT_MODEL(config)
    pt_model.eval()
    print("TensorFlow\n\n")
    tf_test_acc, tf_model = TF_MODEL(config)
    print("MXNet\n\n")
    mx_test_acc, mx_model = MX_MODEL(config)
    # now run attacks
    search_results = {'pt_test_acc': pt_test_acc, 'tf_test_acc': tf_test_acc, 'mx_test_acc': mx_test_acc}
    for attack_type in ['uniform', 'gaussian', 'saltandpepper']:
        for model_type in ['pt', 'tf', 'mx']:
            if model_type == 'pt':
                acc = model_attack(pt_model, model_type, attack_type, config)
            elif model_type == "tf":
                acc = model_attack(tf_model, model_type, attack_type, config)
            else:
                acc = model_attack(mx_model, model_type, attack_type, config)
            search_results[model_type + "_" + attack_type + "_" + "accuracy"] = acc
    all_results = list(search_results.values())
    average_res = float(statistics.mean(all_results))
    search_results['average_res'] = average_res
    tune.report(**search_results)
    return search_results

if __name__ == "__main__":
    print("Warning: running with MXNet in addition to TensorFlow and PyTorch.")
    parser = argparse.ArgumentParser("Start MNIST tuning with hyperspace, specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    args = parser.parse_args()
    if not args.model:
        print("NOTE: Defaulting to MNIST model training...")
    else:
        if args.model == "alexnet_cifar100":
            PT_MODEL = pytorch_alexnet.cifar_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar_tf_objective
            MX_MODEL = mxnet_alexnet.cifar_mxnet_objective
            NUM_CLASSES = 1000
        ## definition of gans as the model type
        elif args.model == "gan":
            pass
        else:
            pass
    # Defining the hyperspace
    hyperparameters = [(0.00001, 0.1),  # learning_rate
                       (0.2, 0.9),  # dropout
                       (10, 100),  # epochs
                       (10, 1000)]  # batch size
    space = create_hyperspace(hyperparameters)

    # Aggregating the results
    results = []
    for section in tqdm(space):
        # create a skopt gp minimize object
        optimizer = Optimizer(section)
        search_algo = SkOptSearch(optimizer, ['learning_rate', 'dropout', 'epochs', 'batch_size'],
                                  metric='average_res', mode='max')
        # not using a gpu because running on local
        try:
            analysis = tune.run(multi_train, search_alg=search_algo, num_samples=1, resources_per_trial={'gpu': 1})
            results.append(analysis)
        except:
            print("Fatal Error in Tune Run: check to make sure that MXNet and Foolbox are actually working.")

    all_pt_results = results[0].results_df
    for i in range(1, len(results)):
        all_pt_results = all_pt_results.append(results[i].results_df)

    all_pt_results.to_csv(args.out)