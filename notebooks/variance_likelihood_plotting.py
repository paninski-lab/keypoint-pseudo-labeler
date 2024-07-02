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


config_file = "./configs/pipeline_example.yaml"  # Update pipeline_example to reflect correct dataset
# load pipeline config file
with open(config_file, "r") as file:
    cfg = yaml.safe_load(file)
    
# load lightning pose config file from the path specified in pipeline config
lightning_pose_config_path = cfg.get("lightning_pose_config")
with open(lightning_pose_config_path, "r") as file:
    lightning_pose_cfg = yaml.safe_load(file)

cfg_lp = DictConfig(lightning_pose_cfg)

data_dir = cfg_lp.data.data_dir
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
input_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/networks/"
        )
    )
data_type = 'lp'
output_df = None
csv_name = "predictions_new.csv"

bodypart_list = cfg_lp["data"]["keypoint_names"]

# Load and format input files and prepare an empty DataFrame for output.
input_dfs, output_df, _ = format_data_walk(input_dir, data_type, csv_name)
print(f'Found {len(input_dfs)} models.')

# Convert list of DataFrames to a 3D NumPy array
data_arrays = [df.to_numpy() for df in input_dfs]
markers_3d_array = np.stack(data_arrays, axis=0)

T = markers_3d_array.shape[1]

# Map keypoint names to keys in input_dfs and crop markers_3d_array
keypoint_is = {}
keys = []
keys_likelihood = []
for i, col in enumerate(input_dfs[0].columns):
    keypoint_is[col] = i
for part in bodypart_list:
    keys.append(keypoint_is[part + '_x'])
    keys.append(keypoint_is[part + '_y'])
    keys.append(keypoint_is[part + '_likelihood'])
    keys_likelihood.append(keypoint_is[part + '_likelihood'])

key_cols = np.array(keys)
likelihood_cols = np.array(keys_likelihood)
likelihoods = markers_3d_array[:, :, likelihood_cols]
markers_3d_array = markers_3d_array[:, :, key_cols]
ensemble_preds, ensemble_vars, keypoints_avg_dict = jax_ensemble(
        markers_3d_array)

# print(f'ensemble_vars, shape={ensemble_vars.shape}: {ensemble_vars}')
# print(f'likelihoods, shape={likelihoods.shape}: {likelihoods}')

likelihood_thresh = 0.9

# Define variance bins using a logarithmic scale
min_var = np.min(ensemble_vars[ensemble_vars > 0])  # Smallest non-zero variance
max_var = np.max(ensemble_vars)
variance_bins = np.logspace(np.log10(min_var), np.log10(max_var), 10)

# Transform likelihoods from error likelihoods to actual likelihoods by subtracting from 1
likelihoods_above_thresh = ((1 - likelihoods) > likelihood_thresh).sum(axis=0)

# Sum the x and y ensemble variances
summed_ensemble_vars = ensemble_vars[:, :, 0] + ensemble_vars[:, :, 1]

# Determine the number of rows needed
num_cols = 3
num_rows = (len(bodypart_list) + num_cols - 1) // num_cols  # Adding 1 for the combined plot

# Initialize figure for subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

# Define a custom colormap from white to blue with power-law normalization
cmap = mcolors.LinearSegmentedColormap.from_list("WhiteBlue", ["white", "blue"])

# Plot for each keypoint
total_heatmap = np.zeros((len(variance_bins) - 1, 6))  # To sum data from all keypoints
for kp_idx, kp in enumerate(bodypart_list):
    heatmap = np.zeros((len(variance_bins) - 1, 6))  # 6 to include counts from 0 to 5

    # Bin the variances and count models with likelihoods above the threshold
    for i in range(len(variance_bins) - 1):
        in_bin = (summed_ensemble_vars[:, kp_idx] >= variance_bins[i]) & (summed_ensemble_vars[:, kp_idx] < variance_bins[i + 1])
        for model_count in range(6):
            heatmap[i, model_count] = np.sum((likelihoods_above_thresh[:, kp_idx] == model_count) & in_bin)
            total_heatmap[i, model_count] += heatmap[i, model_count]  # Sum data from all keypoints

    # Plot the heatmap for each keypoint
    row = kp_idx // num_cols
    col = kp_idx % num_cols
    vmax_value = np.max(heatmap) * 0.75
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_value)
    im = axs[row, col].imshow(heatmap.T, aspect='auto', origin='lower', extent=[0, len(variance_bins) - 1, 0, 6], cmap=cmap, norm=norm)
    axs[row, col].set_xlabel('Ensemble Variance (Log Scale)')
    axs[row, col].set_ylabel(f'Number of Models (Likelihood > {likelihood_thresh})')
    axs[row, col].set_title(f'Keypoint: {kp}')
    
    # Annotate each square with the count value
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            axs[row, col].text(i + 0.5, j + 0.5, f'{heatmap[i, j]:.0f}', ha='center', va='center', color='black')

    # Set the x-axis to log scale and customize ticks for each order of magnitude
    x_ticks = np.arange(len(variance_bins) - 1)
    x_labels = [f'({variance_bins[i]:.3f},\n{variance_bins[i+1]:.3f})' for i in range(len(variance_bins) - 1)]
    axs[row, col].set_xticks(x_ticks + 0.5)
    axs[row, col].set_xticklabels(x_labels, rotation=90)
    axs[row, col].set_yticks(np.arange(6) + 0.5)  # Center y-ticks
    axs[row, col].set_yticklabels(np.arange(6))  # Label y-ticks from 0 to 5

    fig.colorbar(im, ax=axs[row, col], orientation='vertical')

# Remove any unused subplots
for i in range(len(bodypart_list), num_rows * num_cols):
    fig.delaxes(axs.flat[i])

# Save the individual keypoint plots
plt.tight_layout()
output_path = os.path.join(script_dir, f'{os.path.basename(data_dir)}_var_likelihood_heatmap.png')
plt.savefig(output_path)

# Save the combined heatmap separately
fig_combined, ax_combined = plt.subplots(figsize=(5, 5))
vmax_combined = np.max(total_heatmap) * 0.75
norm_combined = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_combined)
im_total = ax_combined.imshow(total_heatmap.T, aspect='auto', origin='lower', extent=[0, len(variance_bins) - 1, 0, 6], cmap=cmap, norm=norm_combined)
ax_combined.set_xlabel('Ensemble Variance (Log Scale)')
ax_combined.set_ylabel(f'Number of Models (Likelihood > {likelihood_thresh})')
ax_combined.set_title(f'Combined Keypoints for {os.path.basename(data_dir)}')

# Annotate each square with the count value
for i in range(total_heatmap.shape[0]):
    for j in range(total_heatmap.shape[1]):
        ax_combined.text(i + 0.5, j + 0.5, f'{total_heatmap[i, j]:.0f}', ha='center', va='center', color='black')

# Set the x-axis to log scale and customize ticks for each order of magnitude
ax_combined.set_xticks(x_ticks + 0.5)
ax_combined.set_xticklabels(x_labels, rotation=90)
ax_combined.set_yticks(np.arange(6) + 0.5)  # Center y-ticks
ax_combined.set_yticklabels(np.arange(6))  # Label y-ticks from 0 to 5

fig_combined.colorbar(im_total, ax=ax_combined, orientation='vertical')

# Save the combined plot
combined_output_path = os.path.join(script_dir, f'{os.path.basename(data_dir)}_var_likelihood_heatmap_combined.png')
fig_combined.tight_layout()
fig_combined.savefig(combined_output_path)