from bi_tune import multi_train, model_attack, bitune_parse_arguments
from simple_mnist import pt_mnist, tf_mnist
from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
from segmentation import pytorch_unet, tensorflow_unet
import thetaspaceray
import argparse

# Default constants
PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True

if __name__ == "__main__":
    """Run experiment with command line arguments."""
    parser = argparse.ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                                     "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    parser.add_argument("-n", "--nodes")
    args = parser.parse_args()
    thetaspaceray.run(args, multi_train)