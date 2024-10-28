import os

import numpy as np
import pandas as pd
import yaml
from eks.core import eks_zscore, jax_ensemble
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
from eks.ibl_pupil_smoother import ensemble_kalman_smoother_ibl_pupil
from eks.utils import (
    convert_lp_dlc,
    convert_slp_dlc,
    make_dlc_pandas_index,
    make_output_dataframe,
    populate_output_dataframe,
)
from omegaconf import DictConfig


def load_cfgs(config_file: str):
    # Load pipeline config file
    with open(config_file, "r") as file:
        cfg = yaml.safe_load(file)

    # Load lightning pose config file from the path specified in pipeline config
    lightning_pose_config_path = cfg.get("lightning_pose_config")
    with open(lightning_pose_config_path, "r") as file:
        lightning_pose_cfg = yaml.safe_load(file)

    cfg_lp = DictConfig(lightning_pose_cfg)
    return cfg, cfg_lp


def find_video_names(data_dir: str, video_directories: list[str]):
    num_videos = 0
    video_names = []
    for video_dir in video_directories:
        video_files = os.listdir(os.path.join(data_dir, video_dir))
        num_videos += len(video_files)
        for video_file in video_files:
            if video_file not in video_names:
                video_names.append(video_file)
    return num_videos, video_names


def format_data_walk(input_dir, data_type, video_name):
    input_dfs_list = []
    keypoint_names = None  # Initialize as None to ensure it's defined correctly later

    # Helper function to traverse directories
    def traverse_directories(directory):
        nonlocal keypoint_names  # Ensure we're using the outer keypoint_names variable
        for root, _, files in os.walk(directory):
            for input_file in files:
                if video_name in input_file:
                    file_path = os.path.join(root, input_file)
                    if data_type == 'slp':
                        markers_curr = convert_slp_dlc(root, input_file)
                        keypoint_names = \
                            [c[1] for c in markers_curr.columns[::3] if not c[1].startswith(
                                'Unnamed')]
                        markers_curr_fmt = markers_curr
                    elif data_type in ['lp', 'dlc']:
                        markers_curr = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
                        keypoint_names = \
                            [c[1] for c in markers_curr.columns[::3] if not c[1].startswith(
                                'Unnamed')]
                        model_name = markers_curr.columns[0][0]
                        if data_type == 'lp':
                            markers_curr_fmt = convert_lp_dlc(
                                markers_curr, keypoint_names, model_name=model_name)
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
    markers_3d_array: np.ndarray,
    bodypart_list: list,
    ensembling_mode: str = 'median',
    zscore_threshold: float = 2
) -> list:
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
        pdindex = make_dlc_pandas_index(
            [bodypart_list[k]],
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


def pipeline_eks(input_csv_names: list, input_dir: str, data_type: str,
                 pseudo_labeler: str, cfg_lp: dict, results_dir: str) -> None:
    bodypart_list = cfg_lp["data"]["keypoint_names"]
    s = None  # optimize s
    s_frames = [(None, None)]  # use all frames for optimization
    output_df = None
    print(input_csv_names)

    for csv_name in input_csv_names:
        # Load and format input files and prepare an empty DataFrame for output.
        input_dfs, output_df, _ = format_data_walk(input_dir, data_type, csv_name)
        print(f'Found {len(input_dfs)} input dfs')
        print(f'Input data for {csv_name} has been read in.')

        # Convert list of DataFrames to a 3D NumPy array
        data_arrays = [df.to_numpy() for df in input_dfs]
        markers_3d_array = np.stack(data_arrays, axis=0)

        # Map keypoint names to keys in input_dfs and crop markers_3d_array
        keypoint_is = {}
        keys = []
        for i, col in enumerate(input_dfs[0].columns):
            keypoint_is[col] = i
        for part in bodypart_list:
            keys.append(keypoint_is[part + '_x'])
            keys.append(keypoint_is[part + '_y'])
            keys.append(keypoint_is[part + '_likelihood'])
        key_cols = np.array(keys)
        markers_3d_array = markers_3d_array[:, :, key_cols]

        df_dicts = None
        if pseudo_labeler == "eks":
            # Call the smoother function
            df_dicts, _ = ensemble_kalman_smoother_singlecam(
                markers_3d_array,
                bodypart_list,
                s,
                s_frames,
                blocks=[]
            )

        elif pseudo_labeler == "eks_pupil":
            df_dicts, _, _ = ensemble_kalman_smoother_ibl_pupil(
                markers_list=markers_3d_array,
                keypoint_names=bodypart_list,
                tracker_name='ensemble-kalman_tracker',
                smooth_params=[0.999, 0.999],
                s_frames=s_frames
            )

        elif pseudo_labeler == "ensemble_mean":
            # Call only ensembling function
            df_dicts = ensemble_only_singlecam(
                markers_3d_array,
                bodypart_list
            )

        # Save eks results in new DataFrames and .csv output files
        for k in range(len(bodypart_list)):
            df = df_dicts[k][bodypart_list[k] + '_df']
            output_df = populate_output_dataframe(df, bodypart_list[k], output_df)
            output_path = os.path.join(results_dir, csv_name)
            output_df.to_csv(output_path)

        print(f"{pseudo_labeler} DataFrame output for {csv_name} successfully converted to CSV."
              f" See at {output_path}")


def collect_missing_eks_csv_paths(video_names: list[str], eks_dir: str) -> list[str]:
    input_csv_names = []
    for video_name in video_names:
        csv_name = video_name.replace(".mp4", ".csv")
        csv_path = os.path.join(eks_dir, csv_name)
        if os.path.exists(csv_path):
            print(f"Post-processed output for {os.path.basename(csv_path)} already exists."
                  f" Skipping.")
        else:
            input_csv_names.append(csv_name)
    return input_csv_names
