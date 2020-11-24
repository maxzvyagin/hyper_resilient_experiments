"""Alternate batched scripting flow for running on multiple nodes on ThetaGPU systems"""

from argparse import ArgumentParser
import sys
import os
from hyperspace import create_hyperspace
import ray
import time
import pickle
import stat

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
    chunk_name = str(chunk).replace(" ", "_")
    chunk_name = chunk_name.replace(", ", "")
    script_name = "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/tmp/scripts/script"+args.out+chunk_name+".sh"
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
    spaces = create_spaces_and_args_pickles(arguments)
    space_chunks = chunks(list(range(len(spaces))), NODES)
    # given these space chunks, run in singularity container on GPU
    for chunk in space_chunks:
        submit_job(chunk, arguments)
