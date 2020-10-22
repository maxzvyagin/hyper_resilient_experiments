### General tuning script to combine all submodules ###
### Only uses PyTorch and TensorFlow ###

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
import sys
import tensorflow as tf
import torch
import torchvision
from torch.utils.data import DataLoader
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm

# Default function definitions
PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective

NUM_CLASSES = 10


def model_attack(model, model_type, attack_type, config):
    if model_type == "pt":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
        if NUM_CLASSES == 100:
            data = DataLoader(torchvision.datasets.CIFAR100("~/datasets/", train=False,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(config['batch_size']))
        else:
            data = DataLoader(torchvision.datasets.MNIST("~/datasets/", train=False,
                                                         transform=torchvision.transforms.ToTensor(),
                                                         target_transform=None, download=True),
                              batch_size=int(config['batch_size']))
        images, labels = [], []
        for sample in data:
            images.append(sample[0].to(device))
            labels.append(sample[1].to(device))
        # images, labels = (torch.from_numpy(images).to(device), torch.from_numpy(labels).to(device))
    elif model_type == "tf":
        fmodel = fb.models.TensorFlowModel(model, bounds=(0, 1))
        if NUM_CLASSES == 100:
            train, test = tfds.load('cifar100', split=['train', 'test'], shuffle_files=False, as_supervised=True)
            data = list(test.batch(config['batch_size']))
        else:
            train, test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, as_supervised=True)
            data = list(test.batch(config['batch_size']))
        images, labels = [], []
        for sample in data:
            fixed_image = (np.array(sample[0]) / 255.0)
            images.append(tf.convert_to_tensor(fixed_image))
            labels.append(sample[1])
    else:
        print("Incorrect model type in model attack. Please try again. Must be either PyTorch or TensorFlow.")
        sys.exit()

    # perform the attacks
    if attack_type == "uniform":
        attack = fb.attacks.L2AdditiveUniformNoiseAttack()
    elif attack_type == "gaussian":
        attack = fb.attacks.L2AdditiveGaussianNoiseAttack()
    elif attack_type == "saltandpepper":
        attack = fb.attacks.SaltAndPepperNoiseAttack()
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
    accuracy_list = []
    print("Performing FoolBox Attacks for " + model_type + " with attack type " + attack_type)
    for i in tqdm(range(len(images))):
        raw_advs, clipped_advs, success = attack(fmodel, images[i], labels[i], epsilons=epsilons)
        if model_type == "pt":
            robust_accuracy = 1 - success.cpu().numpy().astype(float).flatten().mean(axis=-1)
        else:
            robust_accuracy = 1 - success.numpy().astype(float).flatten().mean(axis=-1)
        accuracy_list.append(robust_accuracy)
    return np.array(accuracy_list).mean()


def multi_train(config):
    config = {'epochs': 1, 'batch_size': 64, 'learning_rate': .001, 'dropout': .5}
    # pt_test_acc, pt_model = PT_MODEL(config)
    # pt_model.eval()
    tf_test_acc, tf_model = TF_MODEL(config)
    # now run attacks
    # search_results = {'pt_test_acc': pt_test_acc, 'tf_test_acc': tf_test_acc}
    # for attack_type in ['uniform', 'gaussian', 'saltandpepper']:
    for attack_type in ['saltandpepper']:
        # for model_type in ['pt', 'tf']:
        for model_type in ['tf']:
            if model_type == 'pt':
                acc = model_attack(pt_model, model_type, attack_type, config)
            else:
                acc = model_attack(tf_model, model_type, attack_type, config)
                sys.exit()
            search_results[model_type + "_" + attack_type + "_" + "accuracy"] = acc
    all_results = list(search_results.values())
    average_res = float(statistics.mean(all_results))
    search_results['average_res'] = average_res
    tune.report(**search_results)
    return search_results


if __name__ == "__main__":
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
            NUM_CLASSES = 1000
        ## definition of gans as the model type
        elif args.model == "gan":
            pass
        else:
            print("\n ERROR: Unknown model type. Please try again. Must be one of: mnist, alexnet_cifar100, or gan.\n")
            sys.exit()
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
        analysis = tune.run(multi_train, search_alg=search_algo, num_samples=1, resources_per_trial={'gpu': 1})
        results.append(analysis)

    all_pt_results = results[0].results_df
    for i in range(1, len(results)):
        all_pt_results = all_pt_results.append(results[i].results_df)

    all_pt_results.to_csv(args.out)
