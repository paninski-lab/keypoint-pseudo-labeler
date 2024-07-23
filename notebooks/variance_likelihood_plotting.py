import os
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eks.core import jax_ensemble
from matplotlib.ticker import LogLocator

from pseudo_labeler.utils import format_data_walk
from pseudo_labeler.evaluation import compute_likelihoods_and_variance, plot_heatmaps

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline example script")
    parser.add_argument('--dir', type=str, required=True, help="Directory containing the input data")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_dir = args.dir
    data_type = 'lp'
    csv_name = "test_vid.csv"

    input_dfs, output_df, _ = format_data_walk(input_dir, data_type, csv_name)
    print(f'Found {len(input_dfs)} models.')
    likelihood_thresh = 0.9
    likelihoods_above_thresh, summed_ensemble_vars, combined_df, bodypart_list = compute_likelihoods_and_variance(input_dfs, likelihood_thresh)
    print(summed_ensemble_vars)
    plot_heatmaps(likelihoods_above_thresh, summed_ensemble_vars, bodypart_list, input_dir, likelihood_thresh)

    # Create a DataFrame from summed_ensemble_vars
    summed_ensemble_vars_df = pd.DataFrame(summed_ensemble_vars, columns=bodypart_list)

    # Define the output file path
    output_file_path = os.path.join(input_dir, "summed_ensemble_vars.csv")

    # Save the DataFrame to a CSV file
    summed_ensemble_vars_df.to_csv(output_file_path, index=False)

    print(f'Saved summed_ensemble_vars to {output_file_path}')
