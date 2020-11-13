### script to analyze initial results
### adds columns to the results csv

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def add_columns(filename):
    results = pd.read_csv(filename)
    results['average_pt_resiliency'] = (results['pt_uniform_accuracy'] + results['pt_gaussian_accuracy'] + results[
        'pt_saltandpepper_accuracy']) / 3
    results['average_tf_resiliency'] = (results['tf_uniform_accuracy'] + results['tf_gaussian_accuracy'] + results[
        'tf_saltandpepper_accuracy']) / 3
    results['resiliency_difference'] = results['average_pt_resiliency'] - results['average_tf_resiliency']

    results['average_test_accuracy'] = (results['pt_test_acc'] + results['tf_test_acc']) / 2
    results['average_resiliency'] = (results['average_pt_resiliency'] + results['average_tf_resiliency']) / 2
    results['test_resiliency_diff'] = results['average_test_accuracy'] - results['average_resiliency']
    results.to_csv(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    args = parser.parse_args()
    add_columns(args.file)
