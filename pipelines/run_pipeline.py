"""Example pipeline script."""

#%%
import argparse
import yaml
from omegaconf import DictConfig
import sys
import os
import glob
import numpy as np
import pandas as pd
from pseudo_labeler.utils import format_data_walk
from pseudo_labeler.train import train, inference_with_metrics
from pseudo_labeler.frame_selection import select_frame_idxs_eks, export_frames
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eks')))

from eks.utils import format_data, populate_output_dataframe
from eks.utils import populate_output_dataframe
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam


"""
Directory layout:
outputs/{dataset}/hand=100_pseudo=1000/
    networks/
        rng{k}
    post-processors/
        {pseudo-labeler}_rng={k[0]}-{k[-1]}
    selected-frames/
        hand=100_rng={k[hand-label_seed]}_pseudo=1000_{pseudo-labeler}_{strategy}_rng={k[0]}-{k[-1]} 
Where k is based on ensemble_seeds in .yaml config
"""

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
    for k in cfg["ensemble_seeds"]:
        
        cfg_lp.data.data_dir = data_dir
        
        # update training seeds
        cfg_lp.training.rng_seed_data_pt = k
        
        # add iteration-specific fields to the config
        #translate number of steps into numbers of epoch
        cfg_lp.training.max_epochs = 10
        cfg_lp.training.min_epochs = 10
        cfg_lp.training.unfreeze_step = 30
        
    #     script_dir = os.path.dirname(os.path.abspath(__file__))
    #     parent_dir = os.path.dirname(script_dir)
        results_dir = os.path.join(parent_dir, (
                f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
                f"pseudo={cfg['n_pseudo_labels']}/networks/rng{k}"
            )
        )
        #TODO: train baseline model, and numers of hand label = 100  + numbers of pseudolabel =1000 +
        # TODO; naming convention: /outputs/{dataset}/ 
        os.makedirs(results_dir, exist_ok=True)
        
        #skip training if found model config.yaml
        model_config_checked = os.path.join(results_dir, "config.yaml")

        if os.path.exists(model_config_checked):
            print(f"config.yaml directory found for rng{k}. Skipping training.") 
            num_videos = 0
            for video_dir in cfg["video_directories"]:
                video_files = os.listdir(os.path.join(data_dir, video_dir))
                num_videos += len(video_files)   

            #TODO; can change all the string after as an asterisk
            checkpoint_pattern = os.path.join(results_dir, "*", "*", "*", "*", "*.ckpt")
            checkpoint_files = glob.glob(checkpoint_pattern)
            if checkpoint_files:
                best_ckpt = checkpoint_files[0]  # Assuming you want the first .ckpt file found
            else:
                best_ckpt = None
            
            data_module = None
            trainer = None
        
        else:
            print(f"No config.yaml found for rng{k}. Training the model.")
            best_ckpt, data_module, trainer = train(
                                                    cfg=cfg_lp, 
                                                    results_dir=results_dir,
                                                    min_steps=cfg["min_steps"],
                                                    max_steps=cfg["max_steps"],
                                                    milestone_steps=cfg["milestone_steps"],
                                                    val_check_interval=cfg["val_check_interval"]
                                                    )                                     

        # # -------------------------------------------------------------------------------------
        # # run inference on all InD/OOD videos and compute unsupervised metrics
        # # -------------------------------------------------------------------------------------

        for video_dir in cfg["video_directories"]:
            video_files = [f for f in os.listdir(os.path.join(data_dir, video_dir)) if f.endswith('.mp4')]
            for video_file in video_files:
                # Determine the path for the inference CSV
                inference_csv = os.path.join(results_dir, "video_preds", video_file.replace(".mp4", ".csv"))
                # Check if the inference CSV already exists
                if os.path.exists(inference_csv):
                    print(f"Inference file {inference_csv} already exists. Skipping inference for {video_file}")
                else:
                    print(f"Running inference for {video_file}")
                    results_df = inference_with_metrics(
                        video_file=os.path.join(data_dir, video_dir, video_file),
                        cfg=cfg_lp,
                        preds_file=inference_csv,
                        ckpt_file=best_ckpt,
                        data_module=data_module,
                        trainer=trainer,
                        metrics=True,
                    )
        
            
    # # -------------------------------------------------------------------------------------
    # # post-process network outputs to generate potential pseudo labels (chosen in the next step)
    # # -------------------------------------------------------------------------------------
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
    
    # Load initial hand labels
    initial_labels = pd.read_csv(os.path.join(data_dir, "CollectedData.csv"), header=[0,1,2], index_col=0)

    frames_per_video = cfg["n_pseudo_labels"] / num_videos
    print(f"Frames per video: {frames_per_video}")

    selected_frame_idxs = []    
    labeled_data_dir = os.path.join(data_dir, "labeled-data") 

    # Initialize new_labels with initial hand labels
    new_labels = initial_labels.copy()

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
            
            # # export frames to labeled data directory
            export_frames(
                video_file = video_path,
                save_dir=os.path.join(labeled_data_dir, os.path.splitext(os.path.basename(video_file))[0]),
                frame_idxs=frame_idxs,
                format="png",
                n_digits=8,
                context_frames=0,
            )
            
            # load video predictions for this particular video from post-processed output
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

            # debugging
            # print("Subselected Predictions:")
            # print(subselected_preds)
            
            new_columns = pd.MultiIndex.from_arrays([
                ['rick'] * len(subselected_preds.columns),
                subselected_preds.columns.get_level_values('bodyparts'),
                subselected_preds.columns.get_level_values('coords')
            ], names=['scorer', 'bodyparts', 'coords'])

            # Assign new column index to subselected_preds
            subselected_preds.columns = new_columns
            # append pseudo labels to hand labels

            # TODO: instead of training this 1 time. train the model 5 times based on the init_ensemble_seeds
            new_labels = pd.concat([new_labels, subselected_preds])

            # debugging
            # print("New Labels:")
            # print(new_labels)
    # print(f"New Labels after processing directory {video_dir}:")
    # print(new_labels)

    # # -------------------------------------------------------------------------------------
    # # Check number of labels and save new labels
    # # -------------------------------------------------------------------------------------
    
    #check that we have the right number of labels; new_labels.shape[0] should equal cfg["n_pseudo_labels"] + 
    initial_label_count = initial_labels.shape[0]
    expected_total_labels = initial_label_count + cfg["n_pseudo_labels"]
    if new_labels.shape[0] != expected_total_labels:
        print(f"Warning: Number of labels ({new_labels.shape[0]}) does not match expected count ({expected_total_labels})")
    else:
        print(f"Label count verified: {new_labels.shape[0]} labels")

    # Output the final new_labels to a CSV file
    new_labels_csv = os.path.join(data_dir, f"UpdatedCollectedData_withPseudoLabels_{cfg['pseudo_labeler']}_{cfg['selection_strategy']}.csv")
    new_labels.to_csv(new_labels_csv)
    print(f"New labels saved to {new_labels_csv}")

    # -------------------------------------------------------------------------------------
    # Train models on expanded dataset
    # -------------------------------------------------------------------------------------

    # new_labels_csv = os.path.join(data_dir, f"UpdatedCollectedData_withPseudoLabels_{cfg['pseudo_labeler']}_{cfg['selection_strategy']}.csv")
    ensemble_seeds = cfg["ensemble_seeds"]
    num_models = len(ensemble_seeds)
    print(f'Training {num_models} models on expanded dataset')

    for k in ensemble_seeds:
        print(f'Current model hand-label selection seed = {k}')

        #TODO: train_and_infer() use all of the input 
            #argument: seed k, config path or actual config file (not sure if it matters), data directory, csv, all the arguemtns from pl.Train should also be pass from this function (min step, max step, eval_check)
            #resulst_dir is an argument for train_and_infer() as well
            # train_and_infer() function, min_step argument is config of min_step
            #video_directory() also need to be passed
            # Big idea: for k in seeed, run this train_and_infer(), which sit inside train.py
            # soon implement a baseline, huge long chunk of code baseline or eks, there sould be clear how we want to section that code off. 

        # Load lightning pose config file
        with open(lightning_pose_config_path, "r") as file:
            lightning_pose_cfg = yaml.safe_load(file)
        cfg_lp = DictConfig(lightning_pose_cfg)
        
        # Update config
        cfg_lp.data.data_dir = data_dir
        cfg_lp.training.rng_seed_data_pt = k
        cfg_lp.data.csv_file = new_labels_csv  # Use the new CSV file with pseudo-labels
        
        # Add iteration-specific fields to the config
        cfg_lp.training.max_epochs = 10
        cfg_lp.training.min_epochs = 10
        cfg_lp.training.unfreeze_step = 30
        
        # Define the output directory
        results_dir = os.path.join(
            parent_dir, (
                f"../outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
                f"pseudo={cfg['n_pseudo_labels']}/selected-frames/rng{k}"
            )
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # Check if model has already been trained
        model_config_checked = os.path.join(results_dir, "config.yaml")

        if os.path.exists(model_config_checked):
            print(f"config.yaml directory found for rng{k}. Skipping training.") 
            checkpoint_pattern = os.path.join(results_dir, "*", "*", "*", "*", "*.ckpt")
            checkpoint_files = glob.glob(checkpoint_pattern)
            if checkpoint_files:
                best_ckpt = checkpoint_files[0]
            else:
                best_ckpt = None
            data_module = None
            trainer = None
        
        else:
            print(f"No config.yaml found for rng{k}. Training the model.")
            best_ckpt, data_module, trainer = train(
                cfg=cfg_lp, 
                results_dir=results_dir,
                min_steps=cfg["min_steps"],
                max_steps=cfg["max_steps"],
                milestone_steps=cfg["milestone_steps"],
                val_check_interval=cfg["val_check_interval"]
            )

        # -------------------------------------------------------------------------------------
        # Run inference on all InD/OOD videos and compute unsupervised metrics
        # -------------------------------------------------------------------------------------
        # Collect input csv names from video directory
        for video_dir in cfg["video_directories"]:
            video_files = os.listdir(os.path.join(data_dir, video_dir))
            for video_file in video_files:
                inference_csv = os.path.join(results_dir, "video_preds", (
                        f"hand={cfg_lp.training.train_frames}_rng={k}_"
                        f"pseudo={cfg['n_pseudo_labels']}_"
                        f"{cfg['pseudo_labeler']}_{cfg['selection_strategy']}_"
                        f"rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
                        f"{video_file.replace('.mp4', '.csv')}"
                    )
                )
                if os.path.exists(inference_csv):
                    print(f"Inference file {inference_csv} already exists. Skipping inference for {video_file}")
                else:
                    print(f"Running inference for {video_file}")
                    results_df = inference_with_metrics(
                        video_file=os.path.join(data_dir, video_dir, video_file),
                        cfg=cfg_lp,
                        preds_file=inference_csv,
                        ckpt_file=best_ckpt,
                        data_module=data_module,
                        trainer=trainer,
                        metrics=True,
                    )
print("Completed training and inference for all models with expanded dataset.")
    
    # # -------------------------------------------------------------------------------------
    # # save out all predictions/metrics in dataframe(s)
    # # -------------------------------------------------------------------------------------
    # # can think about this later


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
