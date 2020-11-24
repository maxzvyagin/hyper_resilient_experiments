"""Alternate batched scripting flow for running on multiple nodes on ThetaGPU systems"""

from argparse import ArgumentParser
import sys
import os
# from simple_mnist import pt_mnist, tf_mnist
# from alexnet_cifar import pytorch_alexnet, tensorflow_alexnet
# from segmentation import pytorch_unet, tensorflow_unet
from hyperspace import create_hyperspace
import ray
import time
import pickle
import stat

PT_MODEL = pt_mnist.mnist_pt_objective
TF_MODEL = tf_mnist.mnist_tf_objective
NUM_CLASSES = 10
TRIALS = 25
NO_FOOL = False
MNIST = True
NODES = 4

# submit a job on cobalt using specific parameters
# touch a script? or is there a way to run it without any ugly string concats?

# construct spaces and pickle in specified location, default of /tmp/mzvyagin/pickled_spaces - check if /tmp/mzvyagin exists first

# collate results into specified output file

# run function from bi_tune using arguments  - basically translate main function from bi_tune.py

# main function - specify how many gpus to run on and how many trials per space needed, in addition to other main args

def submit_job(chunk, args):
    command = "qsub -n 1 -A CVD-Mol-AI -t 12:00:00 --attrs pubnet=true "
    script_name = "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/scripts/script"+args.out+str(chunk)+".sh"
    command = command + script_name
    f = open(script_name, "w")
    f.write("singularity shell --nv -B /lus:/lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.10-py3.simg\n")
    f.write("python /home/mzvyagin/hyper_resilient/theta_batch.py -n "+str(chunk)+"\n")
    f.close()
    st = os.stat(script_name)
    os.chmod(script_name, st.st_mode | stat.S_IEXEC)
    os.system(command)

def create_spaces_and_args_pickles(args):
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/hyperres_pickled_args", "wb")
    pickle.dump(args, f)
    f.close()
    print("Dumped arguments to /lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/hyperres_pickled_args, "
          "now creating hyperspaces.")
    # Defining the hyperspace
    if args.model == "segmentation_cityscapes":
        hyperparameters = [(0.00001, 0.1),  # learning_rate
                           (10, 100),  # epochs
                           (8, 24),  # batch size
                           (1, .00000001)]  # epsilon for Adam optimizer
    elif args.model == "segmentation_gis":
        hyperparameters = [(0.00001, 0.1),  # learning_rate
                           (10, 100),  # epochs
                           (100, 1000),  # batch size
                           (1, .00000001)]  # epsilon for Adam optimizer
    else:
        hyperparameters = [(0.00001, 0.1),  # learning_rate
                           (0.2, 0.9),  # dropout
                           (10, 100),  # epochs
                           (10, 500),  # batch size
                           (.00000001, 1)]  # epsilon for Adam optimizer
    # create pickled space
    space = create_hyperspace(hyperparameters)
    f = open("/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/hyperres_pickled_spaces", "wb")
    pickle.dump(space, f)
    f.close()
    print("Dumped scikit opt spaces to /lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/hyperres_pickled_spaces.... "
          "Submitting batch jobs to Cobalt now.")
    return space

def chunks(l, n):
    """ Given a list l, split into equal chunks of length n"""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process_args(args):
    """Setting global variables using arguments"""
    global PT_MODEL, TF_MODEL, NUM_CLASSES, NO_FOOL, NODES, MNIST
    if not args.model:
        print("NOTE: Defaulting to MNIST model training...")
        args.model = "mnist"
    else:
        if args.model == "alexnet_cifar100":
            PT_MODEL = pytorch_alexnet.cifar_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar_tf_objective
            NUM_CLASSES = 100
        ## definition of gans as the model type
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
        elif args.model == "cifar_nofool":
            NO_FOOL = True
            PT_MODEL = pytorch_alexnet.cifar_pt_objective
            TF_MODEL = tensorflow_alexnet.cifar_tf_objective
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
    if not args.nodes:
        print("NOTE: Defaulting to 4 nodes per space...")
    else:
        NODES = int(args.nodes)

if __name__ == "__main__":
    print("WARNING: default file locations are used to pickle arguments and hyperspaces. "
          "DO NOT RUN MORE THAN ONE EXPERIMENT AT A TIME.")
    print("Creating spaces.")
    parser = ArgumentParser("Start ThetaGPU bi_tune run using specified model, out file, number of trials, and number of batches.")
    startTime = time.time()
    ray.init()
    parser.add_argument("-o", "--out")
    parser.add_argument("-m", "--model")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-n", "--nodes", help="Number of GPU nodes to submit on.")
    arguments = parser.parse_args()
    process_args(arguments)
    spaces = create_spaces_and_args_pickles(arguments)
    space_chunks = chunks(list(range(spaces)), NODES)
    # given these space chunks, run in singularity container on GPU
    for chunk in space_chunks:
        submit_job(chunk, arguments)
