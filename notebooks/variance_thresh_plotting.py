import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eks.core import jax_ensemble
from matplotlib.ticker import LogLocator

from pseudo_labeler.utils import format_data_walk

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
input_dir = os.path.join(parent_dir, (f"../outputs/crim13/hand=100_pseudo=1000/networks/"))
data_type = 'lp'
output_df = None
csv_name = "predictions_new.csv"
bodypart_list = ['black_mouse_nose', 'black_mouse_right_ear', 'black_mouse_left_ear', 'black_mouse_top_of_neck',
'black_mouse_right_rear_knee', 'black_mouse_left_rear_knee', 'black_mouse_base_of_tail',
'white_mouse_nose', 'white_mouse_right_ear', 'white_mouse_left_ear', 'white_mouse_top_of_neck',
'white_mouse_right_rear_knee', 'white_mouse_left_rear_knee', 'white_mouse_base_of_tail']

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
        in_bin = (ensemble_vars[:, kp_idx, 0] >= variance_bins[i]) & (ensemble_vars[:, kp_idx, 0] < variance_bins[i + 1])
        for model_count in range(6):
            heatmap[i, model_count] = np.sum((likelihoods_above_thresh[:, kp_idx] == model_count) & in_bin)
            total_heatmap[i, model_count] += heatmap[i, model_count]  # Sum data from all keypoints

    # Plot the heatmap for each keypoint
    row = kp_idx // num_cols
    col = kp_idx % num_cols
    vmax_value = np.max(heatmap) * 0.75
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_value)
    im = axs[row, col].imshow(heatmap.T, aspect='auto', origin='lower', extent=[np.log10(min_var), np.log10(max_var), 0, 5], cmap=cmap, norm=norm)
    axs[row, col].set_xlabel('Ensemble Variance (Log Scale)')
    axs[row, col].set_ylabel(f'Number of Models (Likelihood > {likelihood_thresh})')
    axs[row, col].set_title(f'Keypoint: {kp}')
    
    # Set the x-axis to log scale and customize ticks for each order of magnitude
    x_ticks = np.arange(np.floor(np.log10(min_var)), np.ceil(np.log10(max_var)) + 1)
    x_labels = [f'$10^{{{int(tick)}}}$' for tick in x_ticks]
    axs[row, col].set_xticks(x_ticks)
    axs[row, col].set_xticklabels(x_labels)

    fig.colorbar(im, ax=axs[row, col], orientation='vertical')

# Remove any unused subplots
for i in range(len(bodypart_list), num_rows * num_cols):
    fig.delaxes(axs.flat[i])

# Save the individual keypoint plots
plt.tight_layout()
output_path = os.path.join(script_dir, 'ensemble_variance_heatmap.png')
plt.savefig(output_path)

# Save the combined heatmap separately
fig_combined, ax_combined = plt.subplots(figsize=(5, 5))
vmax_combined = np.max(total_heatmap) * 0.75
norm_combined = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_combined)
im_total = ax_combined.imshow(total_heatmap.T, aspect='auto', origin='lower', extent=[np.log10(min_var), np.log10(max_var), 0, 5], cmap=cmap, norm=norm_combined)
ax_combined.set_xlabel('Ensemble Variance (Log Scale)')
ax_combined.set_ylabel(f'Number of Models (Likelihood > {likelihood_thresh})')
ax_combined.set_title('Combined Keypoints')
ax_combined.set_xticks(x_ticks)
ax_combined.set_xticklabels(x_labels)
fig_combined.colorbar(im_total, ax=ax_combined, orientation='vertical')

# Save the combined plot
combined_output_path = os.path.join(script_dir, 'ensemble_variance_heatmap_combined.png')
fig_combined.tight_layout()
fig_combined.savefig(combined_output_path)