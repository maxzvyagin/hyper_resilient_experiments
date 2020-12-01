import sys
from simple_mnist import pt_mnist, tf_mnist
from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from segmentation import pytorch_unet, tensorflow_unet
import argparse
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
from segmentation.gis_preprocess import pt_gis_train_test_split, tf_gis_test_train_split
from segmentation.tensorflow_unet import get_cityscapes
import spaceray

# Default constants
PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True


def model_attack(model, model_type, attack_type, config):
    if model_type == "pt":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
        # cifar
        if NUM_CLASSES == 100:
            data = DataLoader(torchvision.datasets.CIFAR100("~/datasets/", train=False,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(config['batch_size']))
        elif NUM_CLASSES == 10 and not MNIST:
            data = DataLoader(torchvision.datasets.CIFAR10("~/datasets/", train=False,
                                                           transform=torchvision.transforms.ToTensor(),
                                                           target_transform=None, download=True),
                              batch_size=int(config['batch_size']))
        # cityscapes
        elif NUM_CLASSES == 30:
            data = DataLoader(torchvision.datasets.Cityscapes(
                "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/", split='train', mode='fine', target_type='semantic',
                transform=torchvision.transforms.ToTensor(),
                target_transform=torchvision.transforms.ToTensor()),
                batch_size=int(config['batch_size']))
        # gis
        elif NUM_CLASSES == 1:
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
        if NUM_CLASSES == 100:
            train, test = tfds.load('cifar100', split=['train', 'test'], shuffle_files=False, as_supervised=True)
            data = list(test.batch(int(config['batch_size'])))
        elif NUM_CLASSES == 10 and not MNIST:
            train, test = tfds.load('cifar10', split=['train', 'test'], shuffle_files=False, as_supervised=True)
            data = list(test.batch(int(config['batch_size'])))
        # cityscapes
        elif NUM_CLASSES == 30:
            (x_train, y_train), (x_test, y_test) = get_cityscapes()
            data = list(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(int(config['batch_size'])))
        # gis
        elif NUM_CLASSES == 1:
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
    elif attack_type == "spatial":
        attack = fb.attacks.SpatialAttack()
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
    """Definition of side by side training of pytorch and tensorflow models, plus optional resiliency testing."""
    pt_test_acc, pt_model = PT_MODEL(config)
    pt_model.eval()
    search_results = {'pt_test_acc': pt_test_acc}
    if not NO_FOOL:
        for attack_type in ['uniform', 'gaussian', 'saltandpepper', 'spatial']:
            pt_acc = model_attack(pt_model, "pt", attack_type, config)
            search_results["pt" + "_" + attack_type + "_" + "accuracy"] = pt_acc
    # to avoid weird CUDA OOM errors
    del pt_model
    torch.cuda.empty_cache()
    tf_test_acc, tf_model = TF_MODEL(config)
    search_results['tf_test_acc'] = tf_test_acc
    if not NO_FOOL:
        for attack_type in ['uniform', 'gaussian', 'saltandpepper', 'spatial']:
            pt_acc = model_attack(tf_model, "tf", attack_type, config)
            search_results["tf" + "_" + attack_type + "_" + "accuracy"] = pt_acc
    # save results
    all_results = list(search_results.values())
    average_res = float(statistics.mean(all_results))
    search_results['average_res'] = average_res
    try:
        tune.report(**search_results)
    except:
        print("Couldn't report Tune results. Continuing.")
        pass
    return search_results

def bitune_parse_arguments(args):
    """Parsing arguments specifically for bi tune experiments"""
    global PT_MODEL, TF_MODEL, NUM_CLASSES, NO_FOOL, MNIST, TRIALS
    if not args.model:
        print("NOTE: Defaulting to MNIST model training...")
        args.model = "mnist"
    else:
        if args.model == "alexnet_cifar100":
            PT_MODEL = pytorch_alexnet.cifar100_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar100_tf_objective
            NUM_CLASSES = 100
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
        elif args.model == "cifar100_nofool":
            NO_FOOL = True
            PT_MODEL = pytorch_alexnet.cifar100_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar100_tf_objective
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

if __name__ == "__main__":
    """Run experiment with command line arguments."""
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    args = parser.parse_args()
    spaceray.run_experiment(args, multi_train)