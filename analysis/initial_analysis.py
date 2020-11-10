### script to analyze initial results

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def analyze(filename):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    args = parser.parse_args()
