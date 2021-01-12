from hyper_resilient_experiments.segmentation.gis_preprocess import pt_gis_train_test_split, tf_gis_test_train_split
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

# Default constants
PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
MODEL_SELECT = "pt"
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True
MAX_DIFF = False
FASHION = False
MIN_RESILIENCY = False

def model_attack(model, model_type, attack_type, config, num_classes=NUM_CLASSES):
    print(num_classes)
    global FASHION
    print(FASHION)
    if model_type == "pt":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
        # cifar
        if num_classes == 100:
            data = DataLoader(torchvision.datasets.CIFAR100("~/datasets/", train=False,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(config['batch_size']))
        elif FASHION:
            data = DataLoader(torchvision.datasets.FashionMNIST("~/datasets/", train=False,
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
        elif FASHION:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            f = lambda i: tf.expand_dims(i, -1)
            x_test = f(x_test)
            data = list(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(int(config['batch_size'])))
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


def double_train(config):
    """Definition of side by side training of pytorch and tensorflow models, plus optional resiliency testing."""
    global NUM_CLASSES, MIN_RESILIENCY, MAX_DIFF, MODEL_SELECT
    pt = False
    if MODEL_SELECT == "pt":
        selected_model = PT_MODEL
        pt = True
    else:
        selected_model = TF_MODEL
    test_acc, model = selected_model(config)
    if pt:
        model.eval()
    first_model_results = [test_acc]
    search_results = {'framework': MODEL_SELECT}
    search_results['test_acc1'] = test_acc
    if not NO_FOOL:
        for attack_type in ['gaussian', 'deepfool']:
            acc = model_attack(model, MODEL_SELECT, attack_type, config, num_classes=NUM_CLASSES)
            search_results[MODEL_SELECT + "_" + attack_type + "_" + "accuracy1"] = acc
            first_model_results.append(acc)
    # to avoid weird CUDA OOM errors
    if pt:
        del model
        torch.cuda.empty_cache()
    test_acc, model = selected_model(config)
    if pt:
        model.eval()
    second_model_results = [test_acc]
    search_results['test_acc2'] = test_acc
    if not NO_FOOL:
        for attack_type in ['gaussian', 'deepfool']:
            acc = model_attack(model, MODEL_SELECT, attack_type, config, num_classes=NUM_CLASSES)
            search_results[MODEL_SELECT + "_" + attack_type + "_" + "accuracy2"] = acc
            second_model_results.append(acc)
    # save results
    first_ave = float(statistics.mean(first_model_results))
    second_ave = float(statistics.mean(second_model_results))
    average_res = abs(first_ave - second_ave)
    search_results['average_res'] = average_res
    try:
        tune.report(**search_results)
    except:
        print("Couldn't report Tune results. Continuing.")
        pass
    return search_results

def bitune_parse_arguments(args):
    """Parsing arguments specifically for bi tune experiments"""
    global PT_MODEL, TF_MODEL, NUM_CLASSES, NO_FOOL, MNIST, TRIALS, MAX_DIFF, FASHION, MIN_RESILIENCY, MODEL_SELECT
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
            PT_MODEL = pytorch_unet.gis_pt_objective
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
            MNIST = False
        elif args.model == "fashion":
            PT_MODEL = fashion_pytorch_alexnet.fashion_pt_objective
            TF_MODEL = fashion_tensorflow_alexnet.fashion_tf_objective
            MNIST = False
            FASHION = True
        else:
            print("\n ERROR: Unknown model type. Please try again. "
                  "Must be one of: mnist, alexnet_cifar100, segmentation_cityscapes, or segmentation_gis.\n")
            sys.exit()
    if not args.trials:
        print("NOTE: Defaulting to 25 trials per scikit opt space...")
    else:
        TRIALS = int(args.trials)

    if args.framework == "pt":
        pass
    elif args.framework == "tf":
        MODEL_SELECT = "tf"
    else:
        print("UKNOWN Model Framework. Please try again. Select pt or tf.")
        sys.exit()

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
    parser.add_argument('-f', '--framework', required=True)
    parser.add_argument('-n', '--start_space')
    args = parser.parse_args()
    bitune_parse_arguments(args)
    # print(PT_MODEL
    if args.on_lambda:
        spaceray.run_experiment(args, double_train, ray_dir="~/raylogs", cpu=8, start_space=int(args.start_space))
    else:
        spaceray.run_experiment(args, double_train, ray_dir="/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/raylogs", cpu=8,
                                start_space=int(args.start_space))