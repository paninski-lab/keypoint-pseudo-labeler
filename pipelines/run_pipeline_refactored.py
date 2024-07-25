import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig

from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
from eks.utils import format_data, populate_output_dataframe
from pseudo_labeler.evaluation import (
    collect_preds,
    compute_ens_mean_median,
    compute_ood_snippet_metrics,
    find_model_dirs,
    process_csv_for_sessions_and_frames,
    run_eks_on_snippets,
    run_inference_on_snippets,
)
from pseudo_labeler.frame_selection import (
    export_frames,
    select_frame_idxs_hand,
    select_frame_idxs_random,
)
from pseudo_labeler.train import inference_with_metrics, train, train_and_infer
from pseudo_labeler.utils import format_data_walk, pipeline_eks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eks')))

def pipeline(config_file: str):

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
    
    # -------------------------------------------------------------------------------------
    # train k supervised models on n hand-labeled frames and compute labeled OOD metrics
    # -------------------------------------------------------------------------------------
    print(f'training {len(cfg["ensemble_seeds"])} baseline models')
    eks_input_csv_names = []  # save for EKS

    num_videos = 0
    for video_dir in cfg["video_directories"]:
        video_files = os.listdir(os.path.join(data_dir, video_dir))
        num_videos += len(video_files)   

    # Set pipeline seed
    np.random.seed(cfg["pipeline_seeds"])

    # Create subsample file
    subsample_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_p={cfg['pipeline_seeds']}.csv"
    unsampled_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_p={cfg['pipeline_seeds']}_unsampled.csv"
    unsampled_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_p={cfg['pipeline_seeds']}_unsampled.csv"
    subsample_dir = os.path.join(parent_dir, f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_pseudo={cfg['n_pseudo_labels']}/")
    os.makedirs(subsample_dir, exist_ok=True)
    subsample_path = os.path.join(subsample_dir, subsample_filename)
    unsampled_path = os.path.join(subsample_dir, unsampled_filename)
    unsampled_path = os.path.join(subsample_dir, unsampled_filename)

    # Load the full dataset and create the initial subsample csv
    # Load the full dataset and create the initial subsample csv
    collected_data = pd.read_csv(os.path.join(data_dir, "CollectedData.csv"), header=[0,1,2])
    initial_subsample = collected_data.sample(n=cfg_lp.training.train_frames)
    initial_subsample.to_csv(subsample_path, index=False)
    print(f"Saved initial subsample hand labels CSV file: {subsample_path}")

    # and also create the unsampled csv
    initial_indices = initial_subsample.index
    unsampled = collected_data.drop(index=initial_indices)
    unsampled.to_csv(unsampled_path, index=False)
    print(f"Saved unsampled hand labels CSV file: {unsampled_path}")

    print(f"Saved initial subsample hand labels CSV file: {subsample_path}")

    # and also create the unsampled csv
    initial_indices = initial_subsample.index
    unsampled = collected_data.drop(index=initial_indices)
    unsampled.to_csv(unsampled_path, index=False)
    print(f"Saved unsampled hand labels CSV file: {unsampled_path}")

    for k in cfg["ensemble_seeds"]:

        # Define the output directory
        results_dir = os.path.join(parent_dir, (
        f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
        f"pseudo={cfg['n_pseudo_labels']}/networks/rng{k}"
        ))
        os.makedirs(results_dir, exist_ok=True)  

        # Set the seed for reproducibility
        np.random.seed(k)

        train_and_infer(
            cfg=cfg.copy(),
            cfg_lp=cfg_lp.copy(),
            k=k,
            data_dir=data_dir,
            results_dir=results_dir,
            min_steps=cfg["min_steps"],
            max_steps=cfg["max_steps"],
            milestone_steps=cfg["milestone_steps"],
            val_check_interval=cfg["val_check_interval"],
            video_directories=cfg["video_directories"],
            inference_csv_detailed_naming=False,
            new_labels_csv = subsample_path # Set to None to use the original csv_file
        )


    # # # -------------------------------------------------------------------------------------
    # # # post-process network outputs to generate potential pseudo labels (chosen in the next step)
    # # # -------------------------------------------------------------------------------------
    pseudo_labeler = cfg["pseudo_labeler"]

    input_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/networks/"
        )
    )
    results_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/post-processors/"
            f"{pseudo_labeler}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
        )
    )
    if os.path.exists(results_dir):
        print(f"\n\n\n\n{pseudo_labeler} directory {results_dir} already exists. Skipping post-processing\n.\n.\n.\n")
    else:
        print(f"Post-Processing Network Outputs using method: {pseudo_labeler}\n.\n.\n.\n")
        os.makedirs(results_dir, exist_ok=True)
        data_type = cfg["data_type"]
        output_df = None

        # Collect input csv names from video directory
        input_csv_names = []
        for video_dir in cfg["video_directories"]:
            video_files = os.listdir(os.path.join(data_dir, video_dir))
            for video_file in video_files:
                csv_name = video_file.replace(".mp4", ".csv")
                if csv_name not in input_csv_names:
                    print(f'Appending: {csv_name} to post-processing input csv list')
                    input_csv_names.append(csv_name)

        if pseudo_labeler == "eks" or pseudo_labeler == "ensemble_mean":
            pipeline_eks(input_csv_names, input_dir, data_type, pseudo_labeler, cfg_lp, results_dir)


    # # -------------------------------------------------------------------------------------
    # # run inference on OOD snippets (if specified) -- using network models
    # # -------------------------------------------------------------------------------------

    dataset_name = os.path.basename(cfg_lp.data.data_dir)
    n_hand_labels = cfg_lp.training.train_frames
    n_pseudo_labels = cfg["n_pseudo_labels"]
    seeds= cfg["ensemble_seeds"]
    keypoint_ensemble_list = cfg_lp.data.keypoint_names

    # where the video snippets are stored
    snippets_dir = f"{data_dir}/videos-for-each-labeled-frame"
    # where the network models are stored
    networks_dir = f"/teamspace/studios/this_studio/outputs/{dataset_name}/hand={n_hand_labels}_pseudo={n_pseudo_labels}/networks"
    # where to save eks/post-processor outputs
    eks_save_dir = f"/teamspace/studios/this_studio/outputs/{dataset_name}/hand={n_hand_labels}_pseudo={n_pseudo_labels}/post-processors/eks_rng={seeds[0]}-{seeds[-1]}/eks_ood_snippets"
    # where model configs are stored
    config_dir = f"/teamspace/studios/this_studio/keypoint-pseudo-labeler/configs/"
    # file name of csv where marker data is stored
    ground_truth_csv = 'CollectedData_new.csv'

    model_dirs_list = find_model_dirs(networks_dir, 'rng')
    print(f"Found {len(model_dirs_list)} network model directories")

    if cfg["ood_snippets"]:
        print(f'Starting OOD snippet analysis for {dataset_name}')
        # Step 1: Run inference on video snippets
        ground_truth_df = pd.read_csv(os.path.join(data_dir, ground_truth_csv), skiprows=2)
        run_inference_on_snippets(model_dirs_list, data_dir, snippets_dir, ground_truth_df)

        # Step 2: Run EKS
        df_eks, dfs_markers = run_eks_on_snippets(snippets_dir, model_dirs_list, eks_save_dir, ground_truth_df, keypoint_ensemble_list)

        # Step 3.1: Collect preds from individual models
        collect_preds(model_dirs_list, snippets_dir)

        # Step 3.2: Compute ens mean and median
        compute_ens_mean_median(model_dirs_list, eks_save_dir, 'ens-mean')
        compute_ens_mean_median(model_dirs_list, eks_save_dir, 'ens-median')

        # Step 4: Compute metrics
        compute_ood_snippet_metrics(config_dir, dataset_name, data_dir, ground_truth_csv, model_dirs_list, eks_save_dir)

    # # -------------------------------------------------------------------------------------
    # # select frames to add to the dataset
    # # -------------------------------------------------------------------------------------

    print(f"Total number of videos: {num_videos}")
    selection_strategy = cfg["selection_strategy"]
    selection_strategy = cfg["selection_strategy"]
    print(
        f'selecting {cfg["n_pseudo_labels"]} pseudo-labels using {cfg["pseudo_labeler"]} '
        f'({selection_strategy} strategy)'
        f'({selection_strategy} strategy)'
    )

    selected_frame_idxs = []    
    labeled_data_dir = os.path.join(data_dir, "labeled-data") 

    # Create a new directory for combined hand labels and pseudo labels
    hand_label_and_pseudo_label_dir = os.path.join(parent_dir, (
        f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
        f"pseudo={cfg['n_pseudo_labels']}/pseudo_label_and_hand_label"
    ))
    os.makedirs(hand_label_and_pseudo_label_dir, exist_ok=True)

    hand_labels = pd.read_csv(subsample_path, header=[0,1,2], index_col=0)
    # Process each ensemble seed
    for k in cfg["ensemble_seeds"]:
        # Initialize seed_labels with hand labels for this seed
        seed_labels = hand_labels.copy()
        frame_idxs = []
        preds_csv_path = None
        
        print(f'Using a {selection_strategy} pseudo-label selection strategy.')

        # Random selection strategy
        if selection_strategy == 'random':
            frames_per_video = int(cfg["n_pseudo_labels"] / num_videos)
            print(f"Frames per video: {frames_per_video}")
            for video_dir in cfg["video_directories"]:
                video_files = os.listdir(os.path.join(data_dir, video_dir))
                for video_file in video_files:         
                    video_path = os.path.join(data_dir, video_dir, video_file)

                    frame_idxs = select_frame_idxs_random(
                        video_file=video_path,
                        n_frames_to_select=frames_per_video,
                        seed=k
                    )

                    base_name = os.path.splitext(os.path.basename(video_file))[0]
                    csv_filename = base_name + ".csv"
                    
                    preds_csv_path = os.path.join(parent_dir, (
                            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
                            f"pseudo={cfg['n_pseudo_labels']}/post-processors/"
                            f"{cfg['pseudo_labeler']}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
                        ),
                        csv_filename
                    )

                    selected_frame_idxs.extend(frame_idxs)
                    
                    frame_idxs = frame_idxs.astype(int)
                    print(f'Selected frame indices: {frame_idxs}')
                    
                    # export frames to labeled data directory
                    export_frames(
                        video_file = video_path,
                        save_dir=os.path.join(labeled_data_dir, os.path.splitext(os.path.basename(video_file))[0]),
                        frame_idxs=frame_idxs,
                        format="png",
                        n_digits=8,
                        context_frames=0,
                    )
                    
                    preds_df = pd.read_csv(preds_csv_path, header=[0,1,2], index_col=0)
                    mask = preds_df.columns.get_level_values("coords").isin(["x", "y"])
                    preds_df = preds_df.loc[:, mask]
                    
                    # subselect the predictions corresponding to frame_idxs
                    subselected_preds = preds_df[preds_df.index.isin(frame_idxs)]

                    def generate_new_index(idx, base_name):
                        return f"labeled-data/{base_name}/img{str(idx).zfill(8)}.png"

                    new_index = [generate_new_index(idx, base_name) for idx in subselected_preds.index]
                    subselected_preds.index = new_index

                    print(f'adjusted: {subselected_preds}')

                    standard_scorer_name = 'standard_scorer'

                    new_columns = pd.MultiIndex.from_arrays([
                        [standard_scorer_name] * len(subselected_preds.columns),
                        subselected_preds.columns.get_level_values('bodyparts'),
                        subselected_preds.columns.get_level_values('coords')
                    ], names=['scorer', 'bodyparts', 'coords'])

                    # Assign new column index to subselected_preds
                    subselected_preds.columns = new_columns

                    # seed_labelsuses the standardized scorer
                    if not seed_labels.empty:
                        seed_labels.columns = pd.MultiIndex.from_arrays([
                            [standard_scorer_name] * len(seed_labels.columns),
                            seed_labels.columns.get_level_values('bodyparts'),
                            seed_labels.columns.get_level_values('coords')
                        ], names=['scorer', 'bodyparts', 'coords'])
                    
                    # append pseudo labels to hand labels for this seed
                    seed_labels = pd.concat([seed_labels, subselected_preds])
        
        # Hand selection strategy
        elif selection_strategy == 'hand':
            frame_idxs = select_frame_idxs_hand(
                hand_labels_csv=unsampled_path,
                n_frames_to_select=cfg["n_pseudo_labels"],
                seed=k
            )
            preds_csv_path = unsampled_path

            frame_idxs = frame_idxs.astype(int)
            print(f'Selected frame indices: {frame_idxs}')
            
            # export frames to labeled data directory
            for video_dir in cfg["video_directories"]:
                video_files = os.listdir(os.path.join(data_dir, video_dir))
                for video_file in video_files:         
                    video_path = os.path.join(data_dir, video_dir, video_file)
                    
                    base_name = os.path.splitext(os.path.basename(video_file))[0]

                    export_frames(
                        video_file = video_path,
                        save_dir=os.path.join(labeled_data_dir, base_name),
                        frame_idxs=frame_idxs,
                        format="png",
                        n_digits=8,
                        context_frames=0,
                    )

            preds_df = pd.read_csv(preds_csv_path, header=[0,1,2], index_col=0)
            mask = preds_df.columns.get_level_values("coords").isin(["x", "y"])
            preds_df = preds_df.loc[:, mask]
            
            # subselect the predictions corresponding to frame_idxs
            subselected_preds = preds_df[preds_df.index.isin(frame_idxs)]
            
            def generate_new_index(idx):
                return f"labeled-data/img{str(idx).zfill(8)}.png"

            new_index = [generate_new_index(idx) for idx in subselected_preds.index]
            subselected_preds.index = new_index

            print(f'adjusted: {subselected_preds}')

            standard_scorer_name = 'standard_scorer'

            new_columns = pd.MultiIndex.from_arrays([
                [standard_scorer_name] * len(subselected_preds.columns),
                subselected_preds.columns.get_level_values('bodyparts'),
                subselected_preds.columns.get_level_values('coords')
            ], names=['scorer', 'bodyparts', 'coords'])

            # Assign new column index to subselected_preds
            subselected_preds.columns = new_columns

            # seed_labels uses the standardized scorer
            if not seed_labels.empty:
                seed_labels.columns = pd.MultiIndex.from_arrays([
                    [standard_scorer_name] * len(seed_labels.columns),
                    seed_labels.columns.get_level_values('bodyparts'),
                    seed_labels.columns.get_level_values('coords')
                ], names=['scorer', 'bodyparts', 'coords'])
            
            # append pseudo labels to hand labels for this seed
            seed_labels = pd.concat([seed_labels, subselected_preds])

        # Export the combined hand labels and pseudo labels for this seed
        combined_csv_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_pseudo={cfg['n_pseudo_labels']}_k={k}.csv"
        combined_csv_path = os.path.join(hand_label_and_pseudo_label_dir, combined_csv_filename)
        seed_labels.to_csv(combined_csv_path)
        print(f"Saved combined hand labels and pseudo labels for seed {k} to {combined_csv_path}")

        # Check number of labels for this seed
        expected_total_labels = cfg_lp.training.train_frames + cfg["n_pseudo_labels"]
        if seed_labels.shape[0] != expected_total_labels:
            print(f"Warning: Number of labels for seed {k} ({seed_labels.shape[0]}) does not match expected count ({expected_total_labels})")
        else:
            print(f"Label count verified for seed {k}: {seed_labels.shape[0]} labels")

        print(f"All combined hand labels and pseudo labels saved in {hand_label_and_pseudo_label_dir}")

        # # -------------------------------------------------------------------------------------
        # # Train models on expanded dataset
        # # -------------------------------------------------------------------------------------

        # Define the results directory for this seed
        results_dir = os.path.join(
            parent_dir, (
                f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
                f"pseudo={cfg['n_pseudo_labels']}/results_aeks_{cfg['selection_strategy']}/rng{k}"
            )
        )
        os.makedirs(results_dir, exist_ok=True)

        # Run train_and_infer with the combined hand labels and pseudo labels
        train_and_infer(
            cfg=cfg.copy(),
            cfg_lp=cfg_lp.copy(),
            k=k,
            data_dir=data_dir,
            results_dir=results_dir,
            min_steps=cfg["min_steps"],
            max_steps=cfg["max_steps"],
            milestone_steps=cfg["milestone_steps"],
            val_check_interval=cfg["val_check_interval"],
            video_directories=cfg["video_directories"],
            inference_csv_detailed_naming=True,
            train_frames=cfg_lp.training.train_frames,# + cfg['n_pseudo_labels'],  # Update total frames
            n_pseudo_labels=cfg['n_pseudo_labels'],
            pseudo_labeler=cfg['pseudo_labeler'],
            selection_strategy=cfg['selection_strategy'],
            ensemble_seed_start=cfg['ensemble_seeds'][0],
            ensemble_seed_end=cfg['ensemble_seeds'][-1],
            new_labels_csv=combined_csv_path  # Use the combined CSV file for this seed
        )

        print(f"Completed training and inference for seed {k} using combined hand labels and pseudo labels")

    print("Completed training and inference for all seeds using expanded datasets")

    # # # -------------------------------------------------------------------------------------
    # # # Run EKS on expanded dataset inferences
    # # # -------------------------------------------------------------------------------------
    pseudo_labeler = 'eks'
    input_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/results_aeks_{cfg['selection_strategy']}/"
        )
    )
    results_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/results_aeks_{cfg['selection_strategy']}/"
            f"{pseudo_labeler}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
        )
    )

    if os.path.exists(results_dir):
        print(f"\n\n\n\n{pseudo_labeler} directory {results_dir} already exists. Skipping post-processing\n.\n.\n.\n")
    else:
        print(f"Post-Processing Network Outputs using method: {pseudo_labeler}\n.\n.\n.\n")
        os.makedirs(results_dir, exist_ok=True)
        data_type = cfg["data_type"]
        output_df = None

        # Collect input csv names from video directory
        input_csv_names = []
        for video_dir in cfg["video_directories"]:
            video_files = os.listdir(os.path.join(data_dir, video_dir))
            for video_file in video_files:
                csv_name = video_file.replace(".mp4", ".csv")
                if csv_name not in input_csv_names:
                    print(f'Appending: {csv_name} to post-processing input csv list')
                    input_csv_names.append(csv_name)

        if pseudo_labeler == "eks" or pseudo_labeler == "ensemble_mean":
            pipeline_eks(input_csv_names, input_dir, data_type, pseudo_labeler, cfg_lp, results_dir)


    # # -------------------------------------------------------------------------------------
    # # run inference on OOD snippets (if specified) -- using network models
    # # -------------------------------------------------------------------------------------

    # where the aeks models are stored
    aeks_dir = (
        f"/teamspace/studios/this_studio/outputs/{dataset_name}/"
        f"hand={n_hand_labels}_pseudo={n_pseudo_labels}/"
        f"results_aeks_{cfg['selection_strategy']}"
    )
    # where to save aeks_eks outputs
    aeks_eks_save_dir = (
        f"/teamspace/studios/this_studio/outputs/{dataset_name}/"
        f"hand={n_hand_labels}_pseudo={n_pseudo_labels}/"
        f"results_aeks_{cfg['selection_strategy']}/"
        f"eks_rng={seeds[0]}-{seeds[-1]}"
    )
    model_dirs_list = find_model_dirs(aeks_dir, 'rng')
    print(f"Found {len(model_dirs_list)} network model directories")

    if cfg["ood_snippets"]:

        # remove eks_rng=0-3
        for directory in model_dirs_list:
            if 'eks' in os.path.basename(directory):
                print(f"aEKS model directory {directory} had 'eks' in it and was removed")
                model_dirs_list.remove(directory)
        
        # Step 1: Run inference on video snippets
        ground_truth_df = pd.read_csv(os.path.join(data_dir, ground_truth_csv), skiprows=2)
        run_inference_on_snippets(model_dirs_list, data_dir, snippets_dir, ground_truth_df)

        # Step 2: Run EKS
        df_eks, dfs_markers = run_eks_on_snippets(snippets_dir, model_dirs_list, aeks_eks_save_dir, ground_truth_df, keypoint_ensemble_list)

        # Step 3.1: Collect preds from individual models
        collect_preds(model_dirs_list, snippets_dir)

        # Step 3.2: Compute ens mean and median
        compute_ens_mean_median(model_dirs_list, aeks_eks_save_dir, 'ens-mean')
        compute_ens_mean_median(model_dirs_list, aeks_eks_save_dir, 'ens-median')

        # Step 4: Compute metrics
        compute_ood_snippet_metrics(config_dir, dataset_name, data_dir, ground_truth_csv, model_dirs_list, aeks_eks_save_dir)
            

if __name__ == "__main__":
    # config_file = "../configs/pipeline_example.yaml"
    # pipeline(config_file)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='absolute path to .yaml configuration file',
        type=str,
    )
    args = parser.parse_args()
    pipeline(args.config)

 