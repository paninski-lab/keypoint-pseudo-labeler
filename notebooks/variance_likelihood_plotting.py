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
    csv_name = "predictions_new.csv"

    input_dfs, output_df, _ = format_data_walk(input_dir, data_type, csv_name)
    print(f'Found {len(input_dfs)} models.')
    likelihood_thresh = 0.9
    likelihoods_above_thresh, summed_ensemble_vars, combined_df, bodypart_list = compute_likelihoods_and_variance(input_dfs, likelihood_thresh)
    plot_heatmaps(likelihoods_above_thresh, summed_ensemble_vars, bodypart_list, input_dir, likelihood_thresh)