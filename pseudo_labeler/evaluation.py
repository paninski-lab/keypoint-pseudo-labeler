"""Functions for plotting and making diagnostic videos."""
import gc
import os

import lightning.pytorch as pl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm
from matplotlib.ticker import LogLocator

from eks.core import jax_ensemble
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
from eks.utils import convert_lp_dlc, make_output_dataframe, populate_output_dataframe
from lightning_pose.data.dali import PrepareDALI
from lightning_pose.utils.io import ckpt_path_from_base_path
from lightning_pose.utils.predictions import (
    PredictionHandler,
    get_cfg_file,
    load_model_from_checkpoint,
    predict_single_video,
)
from lightning_pose.utils.scripts import (
    compute_metrics,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)
from pseudo_labeler.utils import format_data_walk


def compute_likelihoods_and_variance(input_dfs, likelihood_thresh=0.9):
    # Convert list of DataFrames to a 3D NumPy array
    data_arrays = [df.to_numpy() for df in input_dfs]
    markers_3d_array = np.stack(data_arrays, axis=0)
    
    # Extract the body parts from the columns
    columns = input_dfs[0].columns
    bodypart_list = set()
    for col in columns:
        if col.endswith('_x'):
            bodypart_list.add(col[:-2])
        elif col.endswith('_y'):
            bodypart_list.add(col[:-2])
        elif col.endswith('_likelihood'):
            bodypart_list.add(col[:-11])

    bodypart_list = list(bodypart_list)  # Convert set to list for consistent ordering
    keys = []
    keys_likelihood = []
    keypoint_is = {col: i for i, col in enumerate(columns)}
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
    
    return likelihoods_above_thresh, summed_ensemble_vars, combined_df, bodypart_list


def plot_heatmaps(likelihoods_above_thresh, summed_ensemble_vars, bodypart_list, input_dir, likelihood_thresh=0.9):
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
                count = np.sum((likelihoods_above_thresh[:, kp_idx] == model_count) & in_bin)
                heatmap[i, model_count] = count
                total_heatmap[i, model_count] += count

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
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f'{os.path.basename(input_dir)}_var_likelihood_heatmap.png')

    plt.savefig(output_path)

    fig_combined, ax_combined = plt.subplots(figsize=(5, 5))
    vmax_combined = np.max(total_heatmap) * 0.75
    norm_combined = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax_combined)
    im_total = ax_combined.imshow(total_heatmap.T, aspect='auto', origin='lower', extent=[0, len(variance_bins) - 1, 0, 6], cmap=cmap, norm=norm_combined)
    ax_combined.set_xlabel('Ensemble Variance (Log Scale)')
    ax_combined.set_ylabel(f'Number of Models (Likelihood > {likelihood_thresh})')
    ax_combined.set_title(f'Combined Keypoints for {os.path.basename(os.path.dirname(os.path.dirname(input_dir)))}')

    for i in range(total_heatmap.shape[0]):
        for j in range(total_heatmap.shape[1]):
            ax_combined.text(i + 0.5, j + 0.5, f'{total_heatmap[i, j]:.0f}', ha='center', va='center', color='black')

    ax_combined.set_xticks(x_ticks + 0.5)
    ax_combined.set_xticklabels(x_labels, rotation=90)
    ax_combined.set_yticks(np.arange(6) + 0.5)
    ax_combined.set_yticklabels(np.arange(6))

    fig_combined.colorbar(im_total, ax=ax_combined, orientation='vertical')
    combined_output_path = os.path.join(script_dir, f'{os.path.basename(input_dir)}_var_likelihood_heatmap_combined.png')
    fig_combined.tight_layout()
    fig_combined.savefig(combined_output_path)


# # ----------------------------
# OOD snippet functions
# # ----------------------------


def find_model_dirs(base_dir, keyword):
    model_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if keyword in dir_name:
                model_dirs.append(os.path.join(root, dir_name))
    return model_dirs


def process_csv_for_sessions_and_frames(ground_truth_df):
    session_frame_info = []
    for idx, row in tqdm(ground_truth_df.iterrows(), total=len(ground_truth_df)):
        first_col_value = row[0]
        path_parts = first_col_value.split('/')
        session = path_parts[1]
        frame = path_parts[2]
        session_frame_info.append((session, frame))
    return session_frame_info


def run_inference_on_snippets(model_dirs_list, data_dir, snippets_dir, ground_truth_df):
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    session_frame_info = process_csv_for_sessions_and_frames(ground_truth_df)
    
    for model_dir in model_dirs_list:
        print(f'Processing model in {os.path.basename(model_dir)}')
        cfg_file = os.path.join(model_dir, "config.yaml")
        model_cfg = DictConfig(yaml.safe_load(open(cfg_file)))
        
        # Update data directories
        model_cfg.data.data_dir = data_dir
        model_cfg.data.video_dir = os.path.join(data_dir, "videos")
        
        # Get model checkpoint
        ckpt_file = ckpt_path_from_base_path(model_dir, model_name=model_cfg.model.model_name)
        
        # Build datamodule
        cfg_new = model_cfg.copy()
        cfg_new.training.imgaug = 'default'
        imgaug_transform = get_imgaug_transform(cfg=cfg_new)
        dataset_new = get_dataset(cfg=cfg_new, data_dir=cfg_new.data.data_dir, imgaug_transform=imgaug_transform)
        datamodule_new = get_data_module(cfg=cfg_new, dataset=dataset_new, video_dir=cfg_new.data.video_dir)
        datamodule_new.setup()
        
        # Load model
        model = load_model_from_checkpoint(cfg=cfg_new, ckpt_file=ckpt_file, eval=True, data_module=datamodule_new)
        model.to("cuda")
        
        model_cfg.dali.base.sequence_length = 16

        for session, frame in tqdm(session_frame_info):
            video_file = os.path.join(snippets_dir, session, frame.replace('png', 'mp4'))
            prediction_csv_file = os.path.join(model_dir, "video_preds_labeled", session, frame.replace('png', 'csv'))
            os.makedirs(os.path.join(model_dir, "video_preds_labeled", session), exist_ok=True)
            
            if not os.path.exists(video_file):
                print(f'Cannot find video snippet for {video_file}. Skipping')
                continue
            if os.path.exists(prediction_csv_file):
                print(f'Prediction csv file already exists for {session}/{frame}. Skipping')
                continue
            
            print(f'{video_file} saved as\n{prediction_csv_file}')
            cfg = get_cfg_file(cfg_file=cfg_new)
            model_type = "context" if cfg.model.model_type == "heatmap_mhcrnn" else "base"
            cfg.training.imgaug = "default"
            
            vid_pred_class = PrepareDALI(
                train_stage="predict",
                model_type=model_type,
                dali_config=cfg.dali,
                filenames=[video_file],
                resize_dims=[cfg.data.image_resize_dims.height, cfg.data.image_resize_dims.width]
            )
            
            # Get loader
            predict_loader = vid_pred_class()
            
            # Initialize prediction handler class
            pred_handler = PredictionHandler(cfg=cfg, data_module=datamodule_new, video_file=video_file)
            
            # Compute predictions
            preds = trainer.predict(model=model, dataloaders=predict_loader, return_predictions=True)
            
            # Process predictions
            preds_df = pred_handler(preds=preds)
            os.makedirs(os.path.dirname(prediction_csv_file), exist_ok=True)
            preds_df.to_csv(prediction_csv_file)
            
            # Clear up memory
            del predict_loader
            del pred_handler
            del vid_pred_class
            gc.collect()
            torch.cuda.empty_cache()
        
        del dataset_new
        del datamodule_new
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_eks_on_snippets(snippets_dir, model_dirs_list, eks_save_dir, ground_truth_df, keypoint_ensemble_list):
    session_frame_info = process_csv_for_sessions_and_frames(ground_truth_df)
    tracker_name = 'heatmap_tracker'
    keypoint_names = None

    # store useful info here
    index_list = []
    results_list = []

    # Step 1: Process and save the CSVs
    for session, frame in session_frame_info:
        # extract all markers
        markers_list = []
        for model_dir in model_dirs_list:
            csv_file = os.path.join(
                model_dir, 'video_preds_labeled', session, frame.replace('png', 'csv'))
            df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
            keypoint_names = [l[1] for l in df_tmp.columns[::3]]
            markers_tmp = convert_lp_dlc(df_tmp, keypoint_names, model_name=tracker_name)
            markers_list.append(markers_tmp)

        dfs_markers = markers_list
        # make empty dataframe to write results into
        df_eks = make_output_dataframe(df_tmp)

        # Convert list of DataFrames to a 3D NumPy array
        data_arrays = [df.to_numpy() for df in markers_list]
        markers_3d_array = np.stack(data_arrays, axis=0)

        # Map keypoint names to keys in input_dfs and crop markers_3d_array
        keypoint_is = {}
        keys = []
        for i, col in enumerate(markers_list[0].columns):
            keypoint_is[col] = i
        for part in keypoint_ensemble_list:
            keys.append(keypoint_is[part + '_x'])
            keys.append(keypoint_is[part + '_y'])
            keys.append(keypoint_is[part + '_likelihood'])
        key_cols = np.array(keys)
        markers_3d_array = markers_3d_array[:, :, key_cols]

        save_dir = os.path.join(eks_save_dir, 'eks', session)
        save_file = os.path.join(save_dir, frame.replace('png', 'csv'))
        if os.path.exists(save_file):
            print(f'Skipping ensembling for {session}/{frame} as it already exists.')
            continue
        else:
            print(f'Ensembling for {session}/{frame}')
            # Call the smoother function
            df_dicts, s_finals = ensemble_kalman_smoother_singlecam(
                markers_3d_array,
                keypoint_ensemble_list,
                smooth_param=None,
                s_frames=[(None, None)],
                blocks=[],
            )
            # put results into new dataframe
            for k, keypoint_name in enumerate(keypoint_ensemble_list):
                keypoint_df = df_dicts[k][keypoint_name + '_df']
                df_eks = populate_output_dataframe(keypoint_df, keypoint_name, df_eks)

            # save eks results
            os.makedirs(save_dir, exist_ok=True)
            df_eks.to_csv(save_file)

    # Step 2: Extract center frame results
    final_predictions_file = os.path.join(eks_save_dir, 'eks', 'predictions_new.csv')
    if os.path.exists(final_predictions_file):
        print(f'Final predictions file {final_predictions_file} already exists. Skipping extraction.')
        return None, None

    print('Extracting center frame results from all sessions.')
    for session, frame in session_frame_info:
        # Construct the path to the saved EKS results
        save_file = os.path.join(eks_save_dir, 'eks', session, frame.replace('png', 'csv'))
        if not os.path.exists(save_file):
            print(f'Missing EKS file for {session}/{frame}. Skipping.')
            continue

        # read csv
        df_eks = pd.read_csv(save_file, header=[0, 1, 2], index_col=0)
        
        # extract result from center frame
        assert df_eks.shape[0] & 2 != 0
        idx_frame = int(np.floor(df_eks.shape[0] / 2))
        frame_file = frame.replace('.mp4', '.png')
        index_name = f'labeled-data/{session}/{frame_file}'
        result = df_eks[df_eks.index == idx_frame].rename(index={idx_frame: index_name})
        results_list.append(result)

    # save final predictions
    results_df = pd.concat(results_list)
    results_df.sort_index(inplace=True)
    # add "set" column so this df is interpreted as labeled data predictions
    results_df.loc[:, ("set", "", "")] = "train"
    results_df.to_csv(os.path.join(eks_save_dir, 'eks', 'predictions_new.csv'))

    return df_eks, dfs_markers


def collect_preds(model_dirs_list, snippets_dir):
    for model_dir in tqdm(model_dirs_list):
        results_file = os.path.join(model_dir, 'video_preds_labeled', 'predictions_new.csv')
        if os.path.exists(results_file):
            print(f'Predictions file {results_file} already exists. Skipping model {model_dir}.')
            continue
        
        results_list = []
        sessions = os.listdir(snippets_dir)
        for session in sessions:
            frames = os.listdir(os.path.join(snippets_dir, session))
            for frame in frames:
                # Load prediction on snippet
                df = pd.read_csv(
                    os.path.join(model_dir, 'video_preds_labeled', session, frame.replace('.mp4', '.csv')), 
                    header=[0, 1, 2], 
                    index_col=0,
                )
                # Extract result from center frame
                assert df.shape[0] & 2 != 0
                idx_frame = int(np.floor(df.shape[0] / 2))
                frame_file = frame.replace('.mp4', '.png')
                index_name = f'labeled-data/{session}/{frame_file}'
                result = df[df.index == idx_frame].rename(index={idx_frame: index_name})
                results_list.append(result)

        # Save final predictions
        results_df = pd.concat(results_list)
        results_df.sort_index(inplace=True)
        # Add "set" column so this df is interpreted as labeled data predictions
        results_df.loc[:, ("set", "", "")] = "train"
        results_df.to_csv(results_file)


def compute_ens_mean_median(model_dirs_list, eks_save_dir, post_processor_type):
    save_dir_ = os.path.join(eks_save_dir, f'{post_processor_type}')
    results_file = os.path.join(save_dir_, 'predictions_new.csv')
    
    if os.path.exists(results_file):
        print(f'Predictions file {results_file} already exists. Skipping computation for {post_processor_type}.')
        return

    markers_list = []
    for model_dir in model_dirs_list:
        csv_file = os.path.join(model_dir, 'video_preds_labeled', 'predictions_new.csv')
        df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
        preds_curr = df_tmp.to_numpy()[:, :-1]  # remove "set" column
        preds_curr = np.delete(preds_curr, list(range(2, preds_curr.shape[1], 3)), axis=1)
        preds_curr = np.reshape(preds_curr, (preds_curr.shape[0], -1, 2))
        markers_list.append(preds_curr[..., None])
    
    # Concatenate across last dim
    pred_arrays = np.concatenate(markers_list, axis=3)
    
    # Compute mean/median across x/y
    if post_processor_type == 'ens-mean':
        ens_mean = np.mean(pred_arrays, axis=3)
    elif post_processor_type == 'ens-median':
        ens_mean = np.median(pred_arrays, axis=3)
    
    ens_likelihood = np.nan * np.zeros((ens_mean.shape[0], ens_mean.shape[1], 1))
    
    # Build dataframe
    xyl = np.concatenate([ens_mean, ens_likelihood], axis=2)
    df_final = pd.DataFrame(
        xyl.reshape(ens_mean.shape[0], -1), 
        columns=df_tmp.columns[:-1],  # remove "set" column, add back in later
        index=df_tmp.index
    )
    df_final.sort_index(inplace=True)
    # Add "set" column so this df is interpreted as labeled data predictions
    df_final.loc[:, ("set", "", "")] = "train"
    os.makedirs(save_dir_, exist_ok=True)
    df_final.to_csv(results_file)


def compute_ood_snippet_metrics(config_dir, dataset_name, data_dir, ground_truth_csv, model_dirs_list, save_dir):
    # Load and prepare the configuration
    cfg_file = os.path.join(config_dir, f"config_{dataset_name}.yaml")
    cfg = DictConfig(yaml.safe_load(open(cfg_file)))
    
    model_cfg = cfg.copy()
    model_cfg.data.data_dir = data_dir
    model_cfg.data.csv_file = ground_truth_csv
    model_cfg.training.imgaug = "default"
    model_cfg.training.train_prob = 1
    model_cfg.training.val_prob = 0
    model_cfg.training.train_frames = 1

    # Initialize dataset and data module
    imgaug_transform = get_imgaug_transform(cfg=model_cfg)
    dataset = get_dataset(cfg=model_cfg, data_dir=model_cfg.data.data_dir, imgaug_transform=imgaug_transform)
    data_module = get_data_module(cfg=model_cfg, dataset=dataset, video_dir=os.path.join(data_dir, dataset_name, "videos"))
    data_module.setup()

    # Compute metrics on aEKS ensemble members
    for model_dir in tqdm(model_dirs_list):
        print(model_dir)
        preds_file = os.path.join(model_dir, 'video_preds_labeled', 'predictions_new.csv')
        metrics_file = os.path.join(model_dir, 'video_preds_labeled', 'metrics_results.csv')
        
        if os.path.exists(metrics_file):
            print(f'Metrics file {metrics_file} already exists. Skipping model {model_dir}.')
            continue
        
        compute_metrics(cfg=model_cfg, preds_file=preds_file, data_module=data_module)

    # Compute metrics on post-processed traces
    post_processor_types = ["ens-mean", "ens-median", "eks"]
    for post_processor_type in post_processor_types:
        print(f'{post_processor_type}')
        preds_file = os.path.join(save_dir, f'{post_processor_type}', 'predictions_new.csv')
        metrics_file = os.path.join(save_dir, f'{post_processor_type}', 'metrics_results.csv')
        
        if os.path.exists(metrics_file):
            print(f'Metrics file {metrics_file} already exists. Skipping post-processor {post_processor_type}.')
            continue
        
        compute_metrics(cfg=model_cfg, preds_file=preds_file, data_module=data_module)


def standardize_scorer_level(df, new_scorer='standard_scorer'):
    """
    Standardizes the 'scorer' level in the MultiIndex to a common name.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to standardize.
    new_scorer : str
        The new name for the 'scorer' level.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the standardized 'scorer' level.
    """
    df.columns = pd.MultiIndex.from_tuples(
        [(new_scorer, bodypart, coord) for scorer, bodypart, coord in df.columns],
        names=df.columns.names
    )
    return df


def compute_ensemble_stddev(
    df_ground_truth,
    df_preds,
    keypoint_ensemble_list,
    scorer_name='standard_scorer'
):
    """
    Parameters
    ----------
    df_ground_truth : List[pd.DataFrame]
        ground truth predictions
    df_preds : List[pd.DataFrame]
        model predictions
    keypoint_ensemble_list : List[str]
        keypoints to include in the analysis

    Returns
    -------
    np.ndarray
        shape (n_frames, n_keypoints)
    """
    # Initial check for NaNs in df_preds
    for i, df in enumerate(df_preds):
        if df.isna().any().any():
            print(f"Warning: NaN values detected in initial DataFrame {i}.")
            nan_indices = df[df.isna().any(axis=1)].index
            nan_columns = df.columns[df.isna().any()]
            print(f"NaN values found at indices: {nan_indices} in columns: {nan_columns}")

    preds = []
    cols_order = None
    for i, df in enumerate(df_preds):
        assert np.all(df.index == df_ground_truth.index), f"Index mismatch between ground truth and predictions at dataframe {i}"
        
        # Standardize the 'scorer' level
        df = standardize_scorer_level(df, scorer_name)
        
        # Remove likelihood columns
        cols_to_keep = [col for col in df.columns if not col[2].endswith('_likelihood') and 'zscore' not in col[2]]
        # Keep only columns matching the keypoint_ensemble_list
        cols_to_keep = [col for col in cols_to_keep if col[1] in keypoint_ensemble_list]
        df = df[cols_to_keep]
        
        print(f"DataFrame {i} kept columns:", df.columns)
        
        # Check for NaNs in the DataFrame
        if df.isna().any().any():
            print(f"Warning: NaN values detected in DataFrame {i} after filtering.")
            nan_indices = df[df.isna().any(axis=1)].index
            nan_columns = df.columns[df.isna().any()]
            print(f"NaN values found at indices: {nan_indices} in columns: {nan_columns}")
        
        # Print the order of the column headers
        if cols_order is None:
            cols_order = df.columns
        else:
            if not (df.columns == cols_order).all():
                print(f"Column order mismatch detected in DataFrame {i}")
                print("Expected order:", cols_order)
                print("Actual order:", df.columns)
                # Ensure bodyparts and coordinates are consistent
                expected_bodyparts_coords = cols_order.droplevel(0).unique()
                actual_bodyparts_coords = df.columns.droplevel(0).unique()
                if not expected_bodyparts_coords.equals(actual_bodyparts_coords):
                    print("Bodyparts and coordinates mismatch detected")
                    print("Expected bodyparts and coordinates:", expected_bodyparts_coords)
                    print("Actual bodyparts and coordinates:", actual_bodyparts_coords)
        
        # Reshape the DataFrame to the appropriate shape
        try:
            arr = df.to_numpy().reshape(df.shape[0], -1, 2)
        except ValueError as e:
            print(f"Reshape error: {e}")
            print(f"DataFrame shape: {df.shape}")
            print(f"Array shape after reshape attempt: {df.to_numpy().shape}")
            raise
        
        preds.append(arr[..., None])
    
    preds = np.concatenate(preds, axis=3)
    
    # Check for NaNs in preds
    if np.isnan(preds).any():
        print("Warning: NaN values detected in preds array.")
        nan_indices = np.argwhere(np.isnan(preds))
        print(f"NaN values found at indices: {nan_indices}")
    else:
        print("No NaN values detected in preds array.")
    
    stddevs = np.std(preds, axis=-1).mean(axis=-1)
    print(f"Stddevs: {stddevs}")
    return stddevs


def compute_percentiles(arr, std_vals, percentiles):
    num_pts = arr[0]
    vals = []
    prctiles = []
    for p in percentiles:
        v = num_pts * p / 100
        idx = np.argmin(np.abs(arr - v))
        # maybe we don't have enough data
        if idx == len(arr) - 1:
            p_ = arr[idx] / num_pts * 100
        else:
            p_ = p
        vals.append(std_vals[idx])
        prctiles.append(p_)
    return vals, prctiles


def cleanaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False)
    ax.tick_params(right=False)


def run_ood_snippets(cfg: Dict, cfg_lp: Dict, data_dir: str, networks_dir: str, pp_dir: str, pseudo_labeler: str):
    """
    Runs the full pipeline for Out-of-Distribution (OOD) snippets, including inference, EKS, prediction collection,
    ensemble computation, and metric computation.

    Args:
    - cfg (dict): Configuration dictionary containing necessary parameters.
    - cfg_lp (dict): Configuration dictionary containing necessary parameters.
    - data_dir (str): Directory containing the data.
    - networks_dir (str): Directory containing network models.
    - pp_dir (str): Directory for storing preprocessed data.
    - pseudo_labeler (str): The labeler used for pseudo-labeling.
    """

    # Extract necessary configuration values
    dataset_name = os.path.basename(cfg_lp.data.data_dir)
    n_hand_labels = cfg["n_hand_labels"]
    n_pseudo_labels = cfg["n_pseudo_labels"]
    seeds = cfg["ensemble_seeds"]
    keypoint_names = cfg_lp.data.keypoint_names

    # Setting directories
    snippets_dir = f"{data_dir}/videos-for-each-labeled-frame"
    pp_ood_dir = os.path.join(pp_dir, f"{pseudo_labeler}_ood_snippets")
    config_dir = "/teamspace/studios/this_studio/keypoint-pseudo-labeler/configs/"
    ground_truth_csv = 'CollectedData_new.csv'

    # Find model directories
    model_dirs_list = find_model_dirs(networks_dir, 'rng')
    print(f"Found {len(model_dirs_list)} network model directories")

    # Step 1: Run inference on video snippets
    ground_truth_df = pd.read_csv(os.path.join(data_dir, ground_truth_csv), skiprows=2)
    run_inference_on_snippets(model_dirs_list, data_dir, snippets_dir, ground_truth_df)

    # Step 2: Run EKS
    df_eks, dfs_markers = run_eks_on_snippets(snippets_dir, model_dirs_list, pp_dir, ground_truth_df, keypoint_names)

    # Step 3.1: Collect preds from individual models
    collect_preds(model_dirs_list, snippets_dir)

    # Step 3.2: Compute ensemble mean and median
    compute_ens_mean_median(model_dirs_list, pp_ood_dir, 'ens-mean')
    compute_ens_mean_median(model_dirs_list, pp_ood_dir, 'ens-median')

    # Step 4: Compute metrics
    compute_ood_snippet_metrics(config_dir, dataset_name, data_dir, ground_truth_csv, model_dirs_list, pp_ood_dir)