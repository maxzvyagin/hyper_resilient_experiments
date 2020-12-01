from bi_tune import model_attack, bitune_parse_arguments, multi_train
from simple_mnist import pt_mnist, tf_mnist
from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from segmentation import pytorch_unet, tensorflow_unet
import argparse
import torch
import spaceray

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

if __name__ == "__main__":
"""Run experiment with command line arguments."""
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    args = parser.parse_args()
    spaceray.run_experiment(args, max_diff_train)
