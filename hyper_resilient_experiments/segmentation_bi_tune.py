"""Implementation of Bi Tune testing for segmentation tasks which cannot be used with normal FoolBox library"""
from hyper_resilient_experiments.segmentation.gis_preprocess import (pt_gis_train_test_split, tf_gis_test_train_split,
                                                                     perturbed_pt_gis_test_data, perturbed_tf_gis_test_data
                                                                     )
import sys
from hyper_resilient_experiments.simple_mnist import pt_mnist, tf_mnist
from hyper_resilient_experiments.alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from hyper_resilient_experiments.segmentation import pytorch_unet, tensorflow_unet
from hyper_resilient_experiments.alexnet_fashion import fashion_pytorch_alexnet, fashion_tensorflow_alexnet
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

import imgaug as ia
import imgaug.augmenters as iaa

# Default constants
PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True
MAX_DIFF = False
FASHION = False
MIN_RESILIENCY = False


def pt_perturbed(dataset):
    """"""

def segmentation_model_attack(model, model_type, attack_type, config, num_classes=NUM_CLASSES):
    """Salt and pepper augmentation of segmentation images, return accuracy - difference between that and normal is
    a measure of resiliency"""

    if model_type == "pt":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
        # cityscapes
        if num_classes == 30:
            data = DataLoader(torchvision.datasets.Cityscapes(
                "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/", split='train', mode='fine', target_type='semantic',
                transform=torchvision.transforms.ToTensor(),
                target_transform=torchvision.transforms.ToTensor()),
                batch_size=int(config['batch_size']))
        # gis
        elif num_classes == 1:
            train_set, test_set = pt_gis_train_test_split()
            data = DataLoader(test_set, batch_size=int(config['batch_size']))
        images, labels = [], []
        for sample in data:
            images.append(sample[0].to(device))
            labels.append(sample[1].to(device))
        images = aug(images)
        # images, labels = (torch.from_numpy(images).to(device), torch.from_numpy(labels).to(device))
    elif model_type == "tf":
        fmodel = fb.models.TensorFlowModel(model, bounds=(0, 1))
        # cityscapes
        if num_classes == 30:
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
        images = aug(images)
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


def segmentation_multi_train(config):
    """Definition of side by side training of pytorch and tensorflow models, plus optional resiliency testing."""
    global NUM_CLASSES, MIN_RESILIENCY, MAX_DIFF
    print(NUM_CLASSES)
    pt_test_acc, pt_model = PT_MODEL(config)
    pt_model.eval()
    search_results = {'pt_test_acc': pt_test_acc}
    if not NO_FOOL:
        for attack_type in ['gaussian', 'deepfool']:
            pt_acc = segmentation_model_attack(pt_model, "pt", attack_type, config, num_classes=NUM_CLASSES)
            search_results["pt" + "_" + attack_type + "_" + "accuracy"] = pt_acc
    # to avoid weird CUDA OOM errors
    del pt_model
    torch.cuda.empty_cache()
    tf_test_acc, tf_model = TF_MODEL(config)
    search_results['tf_test_acc'] = tf_test_acc
    if not NO_FOOL:
        for attack_type in ['gaussian', 'deepfool']:
            pt_acc = model_attack(tf_model, "tf", attack_type, config, num_classes=NUM_CLASSES)
            search_results["tf" + "_" + attack_type + "_" + "accuracy"] = pt_acc
    # save results
    if not MAX_DIFF:
        all_results = list(search_results.values())
        average_res = float(statistics.mean(all_results))
    elif MIN_RESILIENCY:
        test_results = []
        resiliency_results = []
        for key, value in search_results.items():
            if "test" in key:
                test_results.append(value)
            else:
                resiliency_results.append(value)
        test_ave = float(statistics.mean(test_results))
        res_ave = float(statistics.mean(resiliency_results))
        average_res = abs(test_ave-res_ave)
    else:
        pt_results = []
        tf_results = []
        for key, value in search_results.items():
            if "pt" in key:
                pt_results.append(value)
            else:
                tf_results.append(value)
        pt_ave = float(statistics.mean(pt_results))
        tf_ave = float(statistics.mean(tf_results))
        average_res = abs(pt_ave-tf_ave)
    search_results['average_res'] = average_res
    try:
        tune.report(**search_results)
    except:
        print("Couldn't report Tune results. Continuing.")
        pass
    return search_results

def bitune_parse_arguments(args):
    """Parsing arguments specifically for bi tune experiments"""
    global PT_MODEL, TF_MODEL, NUM_CLASSES, NO_FOOL, MNIST, TRIALS, MAX_DIFF, FASHION, MIN_RESILIENCY
    if args.model == "segmentation_cityscapes":
        PT_MODEL = pytorch_unet.cityscapes_pt_objective
        TF_MODEL = tensorflow_unet.cityscapes_tf_objective
        NUM_CLASSES = 30
    elif args.model == "segmentation_gis":
        PT_MODEL = pytorch_unet.gis_pt_objective
        TF_MODEL = tensorflow_unet.gis_tf_objective
        NUM_CLASSES = 1
    else:
        print("\n ERROR: Unknown model type. Please try again. "
              "Must be one of: mnist, alexnet_cifar100, segmentation_cityscapes, or segmentation_gis.\n")
        sys.exit()
    if not args.trials:
        print("NOTE: Defaulting to 25 trials per scikit opt space...")
    else:
        TRIALS = int(args.trials)

    if args.max_diff:
        MAX_DIFF = True
        print("NOTE: Training using Max Diff approach")

    if args.minimize_resiliency:
        MIN_RESILIENCY = True
        print("NOTE: Training using Min Resiliency approach")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    parser.add_argument('-d', "--max_diff", action="store_true")
    parser.add_argument('-r', '--minimize_resiliency', action="store_true")
    parser.add_argument('-l', '--on_lambda', action="store_true")
    args = parser.parse_args()
    bitune_parse_arguments(args)
    # print(PT_MODEL)
    if args.on_lambda:
        spaceray.run_experiment(args, segmentation_multi_train, ray_dir="~/raylogs", cpu=8)
    else:
        spaceray.run_experiment(args, segmentation_multi_train, ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/raylogs", cpu=8)