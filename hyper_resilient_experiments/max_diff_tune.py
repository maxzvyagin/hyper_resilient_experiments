from hyper_resilient_experiments.bi_tune import (model_attack, bitune_parse_arguments, multi_train, PT_MODEL, TF_MODEL,
                                                 NUM_CLASSES, TRIALS, NO_FOOL, MNIST)
from simple_mnist import pt_mnist, tf_mnist
from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from segmentation import pytorch_unet, tensorflow_unet
import argparse
import torch
import spaceray
from ray import tune

global NUM_CLASSES
NUM_CLASSES = 100

def max_diff_train(config):
    pt_test_acc, pt_model = pytorch_alexnet.cifar100_pt_objective(config)
    pt_model.eval()
    search_results = {'pt_test_acc': pt_test_acc}
    pt_test_results = [pt_test_acc]
    attack_type = 'deepfool'
    pt_acc = model_attack(pt_model, "pt", attack_type, config, num_classes=100)
    search_results["pt" + "_" + attack_type + "_" + "accuracy"] = pt_acc
    pt_test_results.append(pt_acc)
    del pt_model
    torch.cuda.empty_cache()
    tf_test_acc, tf_model = tensorflow_alexnet.cifar100_tf_objective(config)
    search_results['tf_test_acc'] = tf_test_acc
    tf_test_results = [tf_test_acc]
    tf_acc = model_attack(tf_model, "tf", attack_type, config, num_classes=100)
    search_results["tf" + "_" + attack_type + "_" + "accuracy"] = tf_acc
    tf_test_results.append(tf_acc)
    # if not NO_FOOL:
    #     for attack_type in ['uniform', 'gaussian', 'saltandpepper', 'spatial']:
    #         tf_acc = model_attack(tf_model, "tf", attack_type, config)
    #         search_results["tf" + "_" + attack_type + "_" + "accuracy"] = tf_acc
    #         tf_test_results.append(tf_acc)
    # take average of each
    pt_ave = sum(pt_test_results)/len(pt_test_results)
    tf_ave = sum(tf_test_results)/len(tf_test_results)
    average_res = abs(pt_ave-tf_ave)
    search_results['average_res'] = average_res
    tune.report(**search_results)
    return search_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    print(NUM_CLASSES)
    args = parser.parse_args()
    bitune_parse_arguments(args)
    spaceray.run_experiment(args, max_diff_train)
