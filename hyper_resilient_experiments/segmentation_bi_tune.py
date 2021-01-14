"""Implementation of Bi Tune testing for segmentation tasks which cannot be used with normal FoolBox library"""
from hyper_resilient_experiments.segmentation.gis_preprocess import (PT_GISDataset,
                                                                     perturbed_pt_gis_test_data, perturbed_tf_gis_test_data
                                                                     )
import sys
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
from pytorch_lightning.metrics import Accuracy

# Default constants
PT_MODEL = pytorch_unet.gis_pt_objective
TF_MODEL = tensorflow_unet.gis_tf_objective
NUM_CLASSES = 1
TRIALS = 25
NO_FOOL = False
MNIST = True
MAX_DIFF = False
FASHION = False
MIN_RESILIENCY = False


def pt_perturbed(dataset):
    """"""

def segmentation_model_attack(model, model_type, config, num_classes=NUM_CLASSES):
    """Salt and pepper augmentation of segmentation images, return accuracy - difference between that and normal is
    a measure of resiliency"""

    if model_type == "pt":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test = perturbed_pt_gis_test_data()
        test_set = PT_GISDataset(test)
        testloader = DataLoader(test_set, batch_size=int(config['batch_size']))
        accuracy = Accuracy()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for sample in tqdm(testloader):
            cuda_in = sample[0].to(device)
            out = model(cuda_in)
            output = out.to('cpu').squeeze(1)
            accuracy(output, sample[1])
            # cuda_in = cuda_in.detach()
            # label = label.detach()
        return accuracy.compute().item()
    elif model_type == "tf":
        x_test, y_test = perturbed_tf_gis_test_data()
        test_acc = model.evaluate(x_test, y_test, batch_size=config['batch_size'])
        return test_acc
    else:
        print("Unknown model type, failure.")
        return None


def segmentation_multi_train(config):
    """Definition of side by side training of pytorch and tensorflow models, plus optional resiliency testing."""
    global NUM_CLASSES, MIN_RESILIENCY, MAX_DIFF
    print(NUM_CLASSES)
    pt_test_acc, pt_model = PT_MODEL(config)
    pt_model.eval()
    search_results = {'pt_test_acc': pt_test_acc}
    if not NO_FOOL:
            pt_acc = segmentation_model_attack(pt_model, "pt", config, num_classes=NUM_CLASSES)
            search_results["pt" + "_" + "noise_attack" + "_" + "accuracy"] = pt_acc
    # to avoid weird CUDA OOM errors
    del pt_model
    torch.cuda.empty_cache()
    tf_test_acc, tf_model = TF_MODEL(config)
    search_results['tf_test_acc'] = tf_test_acc
    if not NO_FOOL:
        tf_acc = segmentation_model_attack(tf_model, "tf", config, num_classes=NUM_CLASSES)
        search_results["tf" + "_" + "noise_attack" + "_" + "accuracy"] = tf_acc
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
        average_res = test_ave-res_ave
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