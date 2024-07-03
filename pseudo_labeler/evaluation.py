"""Functions for plotting and making diagnostic videos."""
import os
import yaml
from omegaconf import DictConfig

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eks.core import jax_ensemble
from matplotlib.ticker import LogLocator

from pseudo_labeler.utils import format_data_walk


def compute_likelihoods_and_variance(cfg_lp, input_dfs, bodypart_list, likelihood_thresh=0.9):
    # Convert list of DataFrames to a 3D NumPy array
    data_arrays = [df.to_numpy() for df in input_dfs]
    markers_3d_array = np.stack(data_arrays, axis=0)
    
    keypoint_is = {col: i for i, col in enumerate(input_dfs[0].columns)}
    keys = []
    keys_likelihood = []
    for part in bodypart_list:
        keys.extend([keypoint_is[part + '_x'], keypoint_is[part + '_y'], keypoint_is[part + '_likelihood']])
        keys_likelihood.append(keypoint_is[part + '_likelihood'])

    key_cols = np.array(keys)
    likelihood_cols = np.array(keys_likelihood)
    likelihoods = markers_3d_array[:, :, likelihood_cols]
    markers_3d_array = markers_3d_array[:, :, key_cols]

    ensemble_preds, ensemble_vars, keypoints_avg_dict = jax_ensemble(markers_3d_array)
    
    likelihoods_above_thresh = (likelihoods > likelihood_thresh).sum(axis=0)
    summed_ensemble_vars = ensemble_vars[:, :, 0] + ensemble_vars[:, :, 1]

    combined_df = pd.DataFrame({'likelihoods_above_thresh': likelihoods_above_thresh.flatten(),
                                'summed_ensemble_vars': summed_ensemble_vars.flatten()})
    
    return likelihoods_above_thresh, summed_ensemble_vars, combined_df


def plot_heatmaps(likelihoods_above_thresh, summed_ensemble_vars, bodypart_list, data_dir, script_dir, likelihood_thresh=0.9):
    min_var = np.min(summed_ensemble_vars[summed_ensemble_vars > 0])
    max_var = np.max(summed_ensemble_vars)
    variance_bins = np.logspace(np.log10(min_var), np.log10(max_var), 10)
    
    num_cols = 3
    num_rows = (len(bodypart_list) + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    cmap = mcolors.LinearSegmentedColormap.from_list("WhiteBlue", ["white", "blue"])
    
    total_heatmap = np.zeros((len(variance_bins) - 1, 6))
    for kp_idx, kp in enumerate(bodypart_list):
        heatmap = np.zeros((len(variance_bins) - 1, 6))
        for i in range(len(variance_bins) - 1):
            in_bin = (summed_ensemble_vars[:, kp_idx] >= variance_bins[i]) & (summed_ensemble_vars[:, kp_idx] < variance_bins[i + 1])
            for model_count in range(6):
                heatmap[i, model_count] = np.sum((likelihoods_above_thresh[:, kp_idx] == model_count) & in_bin)
                total_heatmap[i, model_count] += heatmap[i, model_count]

        row = kp_idx // num_cols
        col = kp_idx % num_cols
        vmax_value = np.max(heatmap) * 0.75
        norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_value)
        im = axs[row, col].imshow(heatmap.T, aspect='auto', origin='lower', extent=[0, len(variance_bins) - 1, 0, 6], cmap=cmap, norm=norm)
        axs[row, col].set_xlabel('Ensemble Variance (Log Scale)')
        axs[row, col].set_ylabel(f'Number of Models (Likelihood > {likelihood_thresh})')
        axs[row, col].set_title(f'Keypoint: {kp}')
        
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                axs[row, col].text(i + 0.5, j + 0.5, f'{heatmap[i, j]:.0f}', ha='center', va='center', color='black')

        x_ticks = np.arange(len(variance_bins) - 1)
        x_labels = [f'({variance_bins[i]:.3f},\n{variance_bins[i+1]:.3f})' for i in range(len(variance_bins) - 1)]
        axs[row, col].set_xticks(x_ticks + 0.5)
        axs[row, col].set_xticklabels(x_labels, rotation=90)
        axs[row, col].set_yticks(np.arange(6) + 0.5)
        axs[row, col].set_yticklabels(np.arange(6))

        fig.colorbar(im, ax=axs[row, col], orientation='vertical')

    for i in range(len(bodypart_list), num_rows * num_cols):
        fig.delaxes(axs.flat[i])

    plt.tight_layout()
    output_path = os.path.join(script_dir, f'{os.path.basename(data_dir)}_var_likelihood_heatmap.png')
    plt.savefig(output_path)

    fig_combined, ax_combined = plt.subplots(figsize=(5, 5))
    vmax_combined = np.max(total_heatmap) * 0.75
    norm_combined = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_combined)
    im_total = ax_combined.imshow(total_heatmap.T, aspect='auto', origin='lower', extent=[0, len(variance_bins) - 1, 0, 6], cmap=cmap, norm=norm_combined)
    ax_combined.set_xlabel('Ensemble Variance (Log Scale)')
    ax_combined.set_ylabel(f'Number of Models (Likelihood > {likelihood_thresh})')
    ax_combined.set_title(f'Combined Keypoints for {os.path.basename(data_dir)}')

    for i in range(total_heatmap.shape[0]):
        for j in range(total_heatmap.shape[1]):
            ax_combined.text(i + 0.5, j + 0.5, f'{total_heatmap[i, j]:.0f}', ha='center', va='center', color='black')

    ax_combined.set_xticks(x_ticks + 0.5)
    ax_combined.set_xticklabels(x_labels, rotation=90)
    ax_combined.set_yticks(np.arange(6) + 0.5)
    ax_combined.set_yticklabels(np.arange(6))

    fig_combined.colorbar(im_total, ax=ax_combined, orientation='vertical')
    combined_output_path = os.path.join(script_dir, f'{os.path.basename(data_dir)}_var_likelihood_heatmap_combined.png')
    fig_combined.tight_layout()
    fig_combined.savefig(combined_output_path)
    