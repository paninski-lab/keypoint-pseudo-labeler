import argparse
import yaml
from omegaconf import DictConfig
import sys
import os
import glob
import numpy as np
import pandas as pd
from pseudo_labeler.utils import format_data_walk
from pseudo_labeler.train import train, inference_with_metrics, train_and_infer
from pseudo_labeler.frame_selection import select_frame_idxs_eks, export_frames
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eks')))

from eks.utils import format_data, populate_output_dataframe
from eks.utils import populate_output_dataframe
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam

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

    for k in cfg["ensemble_seeds"]:

        # Define the output directory
        results_dir = os.path.join(parent_dir, (
        f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
        f"pseudo={cfg['n_pseudo_labels']}/networks/rng{k}"
        ))
        os.makedirs(results_dir, exist_ok=True)  

        # Define a new directory for hand-labeled data CSV files
        hand_label_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/hand_label"
        ))

        os.makedirs(hand_label_dir, exist_ok=True)
        # Skipping the first 3 rows
        collected_data = pd.read_csv(os.path.join(data_dir, "CollectedData.csv"), header=[0,1,2])
        print(f"Loaded CollectedData.csv with {len(collected_data)} rows")

        # Set the seed for reproducibility
        np.random.seed(k)

        # Randomly select distinct rows
        n_hand_labels = cfg_lp.training.train_frames
        selected_indices = np.random.choice(len(collected_data), n_hand_labels, replace=False)
        selected_data = collected_data.iloc[selected_indices]
        print(f"Selected {len(selected_data)} rows for k={k}")

        # Create the new CSV filename
        new_csv_filename = f"CollectedData_hand={n_hand_labels}_k={k}.csv"
        new_csv_path = os.path.join(hand_label_dir, new_csv_filename)

        # Save the selected data to the new CSV file, including the original header
        selected_data.to_csv(new_csv_path, index=False)
        print(f"Saved new CSV file: {new_csv_path}")

        train_and_infer(
            cfg=cfg,
            cfg_lp=cfg_lp,
            k=k,
            data_dir=data_dir,
            results_dir=results_dir,
            min_steps=cfg["min_steps"],
            max_steps=cfg["max_steps"],
            milestone_steps=cfg["milestone_steps"],
            val_check_interval=cfg["val_check_interval"],
            video_directories=cfg["video_directories"],
            inference_csv_detailed_naming=False,
            new_labels_csv = new_csv_path # Set to None to use the original csv_file
        )

    # # # -------------------------------------------------------------------------------------
    # # # post-process network outputs to generate potential pseudo labels (chosen in the next step)
    # # # -------------------------------------------------------------------------------------
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    input_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/networks/"
        )
    )
    results_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/post-processors/"
            f"{cfg['pseudo_labeler']}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
        )
    )
    if os.path.exists(results_dir):
        print(f"\n\n\n\nPost-Processing directory {os.path.basename(results_dir)} already exists. Skipping post-processing\n.\n.\n.\n")
    else:
        print(f"Post-Processing Network Outputs using method: {cfg['pseudo_labeler']}\n.\n.\n.\n")
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

        if cfg["pseudo_labeler"] == "eks":
            bodypart_list = cfg_lp["data"]["keypoint_names"]
            s = None  # optimize s
            s_frames = [(None, None)] # use all frames for optimization
            for csv_name in input_csv_names:
                # Load and format input files and prepare an empty DataFrame for output.
                input_dfs, output_df, _ = format_data_walk(input_dir, data_type, csv_name)
                print(f'Found {len(input_dfs)} input dfs')
                print(f'Input data for {csv_name} has been read into EKS.')

                '''
                This region should be identical to EKS singlecam script
                / will eventually refactor as an EKS func
                '''
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

                # Call the smoother function
                df_dicts, s_finals = ensemble_kalman_smoother_singlecam(
                    markers_3d_array,
                    bodypart_list,
                    s,
                    s_frames,
                    blocks=[],
                    use_optax=True
                )
                ''' end of identical region '''

                # Save eks results in new DataFrames and .csv output files
                for k in range(len(bodypart_list)):
                    df = df_dicts[k][bodypart_list[k] + '_df']
                    output_df = populate_output_dataframe(df, bodypart_list[k], output_df)
                    output_path = os.path.join(results_dir, csv_name)
                    output_df.to_csv(output_path)

                print(f"EKS DataFrame output for {csv_name} successfully converted to CSV. See at {output_path}")

            else:
                output_df = input_dir
            # other baseline pseudolaber implementation

    # # -------------------------------------------------------------------------------------
    # # select frames to add to the dataset
    # # -------------------------------------------------------------------------------------

    print(f"Total number of videos: {num_videos}")

    print(
        f'selecting {cfg["n_pseudo_labels"]} pseudo-labels using {cfg["pseudo_labeler"]} '
        f'({cfg["selection_strategy"]} strategy)'
    )

    frames_per_video = cfg["n_pseudo_labels"] / num_videos
    print(f"Frames per video: {frames_per_video}")

    selected_frame_idxs = []    
    labeled_data_dir = os.path.join(data_dir, "labeled-data") 

    # Create a new directory for combined hand labels and pseudo labels
    hand_label_and_pseudo_label_dir = os.path.join(parent_dir, (
        f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
        f"pseudo={cfg['n_pseudo_labels']}/pseudo_label_and_hand_label"
    ))
    os.makedirs(hand_label_and_pseudo_label_dir, exist_ok=True)

    # Process each ensemble seed
    for k in cfg["ensemble_seeds"]:
     
        # Load hand labels for this seed
        hand_label_dir = os.path.join(parent_dir, (
            f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/hand_label"
        ))
        new_csv_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_k={k}.csv"
        new_csv_path = os.path.join(hand_label_dir, new_csv_filename)
        
        hand_labels = pd.read_csv(new_csv_path, header=[0,1,2], index_col=0)
        
        # Initialize seed_labels with hand labels for this seed
        seed_labels = hand_labels.copy()

        for video_dir in cfg["video_directories"]:
            video_files = os.listdir(os.path.join(data_dir, video_dir))
            for video_file in video_files:         
                video_path = os.path.join(data_dir, video_dir, video_file)
                frame_idxs = select_frame_idxs_eks(
                    video_file=video_file,
                    n_frames_to_select=frames_per_video,
                )
                selected_frame_idxs.extend(frame_idxs)
                
                frame_idxs = frame_idxs.astype(int)
                
                # export frames to labeled data directory
                export_frames(
                    video_file = video_path,
                    save_dir=os.path.join(labeled_data_dir, os.path.splitext(os.path.basename(video_file))[0]),
                    frame_idxs=frame_idxs,
                    format="png",
                    n_digits=8,
                    context_frames=0,
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
                
                preds_df = pd.read_csv(preds_csv_path, header=[0,1,2], index_col=0)
                mask = preds_df.columns.get_level_values("coords").isin(["x", "y"])
                preds_df = preds_df.loc[:, mask]
                
                # subselect the predictions corresponding to frame_idxs
                subselected_preds = preds_df[preds_df.index.isin(frame_idxs)]

                def generate_new_index(idx, base_name):
                    return f"labeled-data/{base_name}/img{str(idx).zfill(8)}.png"

                new_index = [generate_new_index(idx, base_name) for idx in subselected_preds.index]
                subselected_preds.index = new_index
                
                new_columns = pd.MultiIndex.from_arrays([
                    ['rick'] * len(subselected_preds.columns),
                    subselected_preds.columns.get_level_values('bodyparts'),
                    subselected_preds.columns.get_level_values('coords')
                ], names=['scorer', 'bodyparts', 'coords'])

                # Assign new column index to subselected_preds
                subselected_preds.columns = new_columns
                
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

    # -------------------------------------------------------------------------------------
    # Train models on expanded dataset
    # -------------------------------------------------------------------------------------

    for k in cfg["ensemble_seeds"]:
        # Define the new CSV file path for this seed
        combined_csv_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_pseudo={cfg['n_pseudo_labels']}_k={k}.csv"
        combined_csv_path = os.path.join(hand_label_and_pseudo_label_dir, combined_csv_filename)

        # Define the results directory for this seed
        results_dir = os.path.join(
            parent_dir, (
                f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
                f"pseudo={cfg['n_pseudo_labels']}/results_amortized_eks/rng{k}"
            )
        )
        os.makedirs(results_dir, exist_ok=True)

        # Run train_and_infer with the combined hand labels and pseudo labels
        train_and_infer(
            cfg=cfg,
            cfg_lp=cfg_lp,
            k=k,
            data_dir=data_dir,
            results_dir=results_dir,
            min_steps=cfg["min_steps"],
            max_steps=cfg["max_steps"],
            milestone_steps=cfg["milestone_steps"],
            val_check_interval=cfg["val_check_interval"],
            video_directories=cfg["video_directories"],
            inference_csv_detailed_naming=False,
            train_frames=cfg_lp.training.train_frames + cfg['n_pseudo_labels'],  # Update total frames
            n_pseudo_labels=cfg['n_pseudo_labels'],
            pseudo_labeler=cfg['pseudo_labeler'],
            selection_strategy=cfg['selection_strategy'],
            ensemble_seed_start=cfg['ensemble_seeds'][0],
            ensemble_seed_end=cfg['ensemble_seeds'][-1],
            new_labels_csv=combined_csv_path  # Use the combined CSV file for this seed
        )

        print(f"Completed training and inference for seed {k} using combined hand labels and pseudo labels")

    print("Completed training and inference for all seeds using expanded datasets")    
    

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

 