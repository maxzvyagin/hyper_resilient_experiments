"""Given a directory of results, concatenate all csvs into a single csv"""
import pandas as pd
import os
from argparse import ArgumentParser

def concatenate_results(dir):
    f = os.listdir(dir)
    first = True
    for file in f:
        if ".csv" in file:
            df = pd.read_csv(dir+"/"+file)
            if first:
                first = False
                full_df = df
            else:
                full_df.append(df)
                print(len(full_df))

    full_df.to_csv(dir+"/all_results.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('results_dir')
    args = parser.parse_args()
    concatenate_results(args.results_dir)
