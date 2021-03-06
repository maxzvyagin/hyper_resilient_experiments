from hyper_resilient_experiments.segmentation.gis_preprocess import pt_gis_train_test_split, tf_gis_test_train_split
import sys
from hyper_resilient_experiments.simple_mnist import pt_mnist, tf_mnist
from hyper_resilient_experiments.alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from hyper_resilient_experiments.segmentation import pytorch_unet, tensorflow_unet
import argparse
import ray
from ray import tune
import statistics
import foolbox as fb
import tensorflow as tf
import torch
import torchvision
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from hyper_resilient_experiments.segmentation.tensorflow_unet import get_cityscapes
import spaceray
import faulthandler

def gis_model_attack(model, model_type, attack_type, config, num_classes=30):
    print(num_classes)
    MNIST = 0
    if model_type == "pt":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
        # cifar
        if num_classes == 100:
            data = DataLoader(torchvision.datasets.CIFAR100("~/datasets/", train=False,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(config['batch_size']))
        elif num_classes == 10 and not MNIST:
            data = DataLoader(torchvision.datasets.CIFAR10("~/datasets/", train=False,
                                                           transform=torchvision.transforms.ToTensor(),
                                                           target_transform=None, download=True),
                              batch_size=int(config['batch_size']))
        # cityscapes
        elif num_classes == 30:
            data = DataLoader(torchvision.datasets.Cityscapes(
                "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/", split='train', mode='fine', target_type='semantic',
                transform=torchvision.transforms.ToTensor(),
                target_transform=torchvision.transforms.ToTensor()),
                batch_size=int(config['batch_size']))
        # gis
        elif num_classes == 1:
            train_set, test_set = pt_gis_train_test_split()
            data = DataLoader(test_set, batch_size=int(config['batch_size']))
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
        # cifar
        if num_classes == 100:
            train, test = tfds.load('cifar100', split=['train', 'test'], shuffle_files=False, as_supervised=True)
            data = list(test.batch(int(config['batch_size'])))
        elif num_classes == 10 and not MNIST:
            train, test = tfds.load('cifar10', split=['train', 'test'], shuffle_files=False, as_supervised=True)
            data = list(test.batch(int(config['batch_size'])))
        # cityscapes
        elif num_classes == 30:
            (x_train, y_train), (x_test, y_test) = get_cityscapes()
            data = list(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(int(config['batch_size'])))
        # gis
        elif num_classes == 1:
            (x_train, y_train), (x_test, y_test) = tf_gis_test_train_split()
            data = list(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(int(config['batch_size'])))
        # mnist
        else:
            train, test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, as_supervised=True)
            data = list(test.batch(int(config['batch_size'])))
        images, labels = [], []
        for sample in data:
            fixed_image = (np.array(sample[0]) / 255.0).astype('float32')
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
    elif attack_type == "boundary":
        attack = fb.attacks.BoundaryAttack()
    # NOTE: Doesn't look like spatial is being supported by the devs anymore, not sure if should be using
    elif attack_type == "spatial":
        attack = fb.attacks.SpatialAttack()
    elif attack_type == "deepfool":
        attack = fb.attacks.LinfDeepFoolAttack()
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


def gis_multi_train(config):
    """Definition of side by side training of pytorch and tensorflow models, plus optional resiliency testing."""
    pt_test_acc, pt_model = pytorch_unet.cityscapes_pt_objective(config)
    pt_model.eval()
    search_results = {'pt_test_acc': pt_test_acc}
    for attack_type in ['gaussian', 'deepfool']:
        pt_acc = gis_model_attack(pt_model, "pt", attack_type, config, num_classes=1)
        search_results["pt" + "_" + attack_type + "_" + "accuracy"] = pt_acc
    # to avoid weird CUDA OOM errors
    del pt_model
    torch.cuda.empty_cache()
    tf_test_acc, tf_model = tensorflow_unet.cityscapes_tf_objective(config)
    search_results['tf_test_acc'] = tf_test_acc
    for attack_type in ['gaussian', 'deepfool']:
        pt_acc = gis_model_attack(tf_model, "tf", attack_type, config, num_classes=1)
        search_results["tf" + "_" + attack_type + "_" + "accuracy"] = pt_acc
    # save results
    all_results = list(search_results.values())
    average_res = float(statistics.mean(all_results))
    # pt_results = []
    # tf_results = []
    # for key, value in search_results.items():
    #     if "pt" in key:
    #         pt_results.append(value)
    #     else:
    #         tf_results.append(value)
    # pt_ave = float(statistics.mean(pt_results))
    # tf_ave = float(statistics.mean(tf_results))
    # average_res = abs(pt_ave-tf_ave)
    search_results['average_res'] = average_res
    try:
        tune.report(**search_results)
    except:
        print("Couldn't report Tune results. Continuing.")
        pass
    return search_results

if __name__ == "__main__":
    test_config = {'batch_size': 1, 'learning_rate': .001, 'epochs': 1, 'adam_epsilon': 10 ** -9}
    gis_multi_train(test_config)
