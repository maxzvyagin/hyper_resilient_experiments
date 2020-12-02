from bi_tune import model_attack, bitune_parse_arguments, multi_train
from simple_mnist import pt_mnist, tf_mnist
from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from segmentation import pytorch_unet, tensorflow_unet
import argparse
import torch
import spaceray
from ray import tune

PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True

def max_diff_train(config):
    pt_test_acc, pt_model = PT_MODEL(config)
    pt_model.eval()
    search_results = {'pt_test_acc': pt_test_acc}
    pt_test_results = [pt_test_acc]
    if not NO_FOOL:
        for attack_type in ['uniform', 'gaussian', 'saltandpepper', 'spatial']:
            pt_acc = model_attack(pt_model, "pt", attack_type, config)
            search_results["pt" + "_" + attack_type + "_" + "accuracy"] = pt_acc
            pt_test_results.append(pt_acc)
    # to avoid weird CUDA OOM errors
    del pt_model
    torch.cuda.empty_cache()
    tf_test_acc, tf_model = TF_MODEL(config)
    search_results['tf_test_acc'] = tf_test_acc
    tf_test_results = [tf_test_acc]
    if not NO_FOOL:
        for attack_type in ['uniform', 'gaussian', 'saltandpepper', 'spatial']:
            tf_acc = model_attack(tf_model, "tf", attack_type, config)
            search_results["tf" + "_" + attack_type + "_" + "accuracy"] = tf_acc
            tf_test_results.append(tf_acc)
    # take average of each
    pt_ave = sum(pt_test_results)/len(pt_test_results)
    tf_ave = sum(tf_test_results)/len(tf_test_results)
    average_res = abs(pt_ave-tf_ave)
    search_results['average_res'] = average_res
    tune.report(**search_results)
    return search_results

if __name__ == "__main__":
"""Run experiment with command line arguments."""
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    args = parser.parse_args()
    bitune_parse_arguments(args)
    print(PT_MODEL)
    print(TF_MODEL)
    spaceray.run_experiment(args, max_diff_train)
