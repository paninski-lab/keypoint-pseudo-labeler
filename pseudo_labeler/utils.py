import os
import pandas as pd
import numpy as np

from eks.utils import convert_slp_dlc, convert_lp_dlc, make_output_dataframe, make_dlc_pandas_index
from eks.core import jax_ensemble, eks_zscore
from eks.singlecam_smoother import adjust_observations


def format_data_walk(input_dir, data_type, video_name):
    input_dfs_list = []
    keypoint_names = None  # Initialize as None to ensure it's defined correctly later

    # Helper function to traverse directories
    def traverse_directories(directory):
        nonlocal keypoint_names  # Ensure we're using the outer keypoint_names variable
        for root, _, files in os.walk(directory):
            for input_file in files:
                if input_file == video_name:
                    file_path = os.path.join(root, input_file)

                    if data_type == 'slp':
                        markers_curr = convert_slp_dlc(root, input_file)
                        keypoint_names = [c[1] for c in markers_curr.columns[::3] if not c[1].startswith('Unnamed')]
                        markers_curr_fmt = markers_curr
                    elif data_type in ['lp', 'dlc']:
                        markers_curr = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
                        keypoint_names = [c[1] for c in markers_curr.columns[::3] if not c[1].startswith('Unnamed')]
                        model_name = markers_curr.columns[0][0]
                        if data_type == 'lp':
                            markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
                        else:
                            markers_curr_fmt = markers_curr

                    # Modify the frame labels to incremented integers starting at 0
                    markers_curr_fmt.index = range(len(markers_curr_fmt))

                    markers_curr_fmt.to_csv('fmt_input.csv', index=False)
                    input_dfs_list.append(markers_curr_fmt)

    # Traverse input_dir and its subdirectories
    traverse_directories(input_dir)

    if len(input_dfs_list) == 0:
        raise FileNotFoundError(f'No file named {video_name} found in {input_dir}')

    output_df = make_output_dataframe(input_dfs_list[0])
    # returns both the formatted marker data and the empty dataframe for EKS output
    return input_dfs_list, output_df, keypoint_names


def ensemble_only_singlecam(
        markers_3d_array, bodypart_list, ensembling_mode='median', zscore_threshold=2):
    """
    Perform ensembling on 3D marker data from a single camera.

    Parameters:
    markers_3d_array (np.ndarray): 3D array of marker data.
    bodypart_list (list): List of body parts.
    ensembling_mode (str): Mode for ensembling ('median' by default).
    zscore_threshold (float): Z-score threshold.

    Returns:
    list: Dataframes with ensembled predictions.
    """

    T = markers_3d_array.shape[1]
    n_keypoints = markers_3d_array.shape[2] // 3

    # Compute ensemble statistics
    print("Ensembling models")
    ensemble_preds, ensemble_vars, keypoints_avg_dict = jax_ensemble(
        markers_3d_array, mode=ensembling_mode)

    dfs = []
    df_dicts = []

    # Process each keypoint
    for k in range(n_keypoints):
        # Computing z-score
        zscore = eks_zscore(ensemble_preds[:, k, :],
                            ensemble_preds[:, k, :],
                            ensemble_vars[:, k, :],
                            min_ensemble_std=zscore_threshold)

        # Final Cleanup
        pdindex = make_dlc_pandas_index([bodypart_list[k]],
                                        labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])
        
        # Extract predictions and variances
        x_vals = ensemble_preds[:, k, 0]
        y_vals = ensemble_preds[:, k, 1]
        x_vars = ensemble_vars[:, k, 0]
        y_vars = ensemble_vars[:, k, 1]

        pred_arr = np.vstack([
            x_vals.T,
            y_vals.T,
            np.full(T, np.nan),  # likelihood is not computed
            x_vars.T,
            y_vars.T,
            np.full(T, zscore)
        ]).T

        df = pd.DataFrame(pred_arr, columns=pdindex)
        dfs.append(df)
        df_dicts.append({bodypart_list[k] + '_df': df})

    return df_dicts