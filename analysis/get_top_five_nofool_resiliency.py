import sys
import pandas as pd
sys.path.append("/home/mzvyagin/hyper_resilient")
from bi_tune import multi_train
import argparse

def test_top_five(i, o):
    ray_results = pd.read_csv(i)
    sorted_ray_results = ray_results.sort_values('average_res', ascending=False)
    sorted_ray_results = sorted_ray_results.reset_index(drop=True)
    learning_rate = ray_results['config.learning_rate']
    dropout = ray_results['config.dropout']
    epochs = ray_results['config.epochs']
    batch_size = ray_results['config.batch_size']
    records = []
    for i in list(range(5)):
        config = {'learning_rate': learning_rate[i], 'dropout': dropout[i], 'epochs': epochs[i],
                  'batch_size': batch_size[i]}
        search_results = multi_train(config)
        search_results['rank'] = i
        records.append(search_results)
    csv = pd.DataFrame.from_records(records)
    csv.to_csv(o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Specify input CSV from results.")
    parser.add_argument('-o', '--output', help="Specify output file name")
    args = parser.parse_args()
    test_top_five(args.input, args.output)