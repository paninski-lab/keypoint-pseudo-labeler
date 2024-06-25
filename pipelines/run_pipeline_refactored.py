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
    # Load lightning pose config file
        # Define the output directory
        results_dir = os.path.join(
            parent_dir, 
            f"../outputs/mirror-mouse/100_1000-{cfg.get('pseudo_labeler', 'eks')}-{cfg.get('selection_strategy', 'random')}/rng{k}"
        )
        os.makedirs(results_dir, exist_ok=True)
        
        train_and_infer(
            cfg = cfg, 
            cfg_lp = cfg_lp, 
            k = k, 
            data_dir = data_dir, 
            results_dir = results_dir, 
            min_steps=cfg["min_steps"],
            max_steps=cfg["max_steps"],
            milestone_steps=cfg["milestone_steps"],
            val_check_interval=cfg["val_check_interval"],
            video_directories=cfg["video_directories"],
            new_labels_csv= None  # Set to None to use the original csv_file
        )

    # # -------------------------------------------------------------------------------------
    # print("Starting EKS")
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(script_dir)
    # results_dir = os.path.join(parent_dir, f"../outputs/mirror-mouse/100_1000-eks-random")
    # input_dir = results_dir
    # os.makedirs(results_dir, exist_ok=True)  # can be removed later (if results_dir exists)
    # eks_dir = os.path.join(results_dir, "eks")
    # os.makedirs(eks_dir, exist_ok=True)  # Ensure eks directory exists
    # data_type = cfg["data_type"]
    # output_df = None

    # if cfg["pseudo_labeler"] == "eks":
    #     bodypart_list = cfg_lp["data"]["keypoint_names"]
    #     s = None  # optimize s
    #     s_frames = [(None, None)] # use all frames for optimization
    #     for csv_name in eks_input_csv_names:
    #         # Load and format input files and prepare an empty DataFrame for output.
    #         input_dfs, output_df, _ = format_data_walk(input_dir, data_type, csv_name)
    #         print(f'Found {len(input_dfs)} input dfs')
    #         print(f'Input data for {csv_name} has been read into EKS.')

    #         ''' This region should be identical to EKS singlecam script '''
    #         # Convert list of DataFrames to a 3D NumPy array
    #         data_arrays = [df.to_numpy() for df in input_dfs]
    #         markers_3d_array = np.stack(data_arrays, axis=0)

    #         # Map keypoint names to keys in input_dfs and crop markers_3d_array
    #         keypoint_is = {}
    #         keys = []
    #         for i, col in enumerate(input_dfs[0].columns):
    #             keypoint_is[col] = i
    #         for part in bodypart_list:
    #             keys.append(keypoint_is[part + '_x'])
    #             keys.append(keypoint_is[part + '_y'])
    #             keys.append(keypoint_is[part + '_likelihood'])
    #         key_cols = np.array(keys)
    #         markers_3d_array = markers_3d_array[:, :, key_cols]

    #         # Call the smoother function
    #         df_dicts, s_finals, nll_values_array = ensemble_kalman_smoother_singlecam(
    #             markers_3d_array,
    #             bodypart_list,
    #             s,
    #             s_frames,
    #             blocks=[],
    #             use_optax=True
    #         )
    #         ''' end of identical region '''

    #         # Save eks results in new DataFrames and .csv output files
    #         for k in range(len(bodypart_list)):
    #             df = df_dicts[k][bodypart_list[k] + '_df']
    #             output_df = populate_output_dataframe(df, bodypart_list[k], output_df)
    #             output_path = os.path.join(eks_dir, csv_name)
    #             output_df.to_csv(output_path)

    #         print(f"EKS DataFrame output for {csv_name} successfully converted to CSV. See at {output_path}")

    #     else:
    #         output_df = input_dir
    # #         # other baseline pseudolaber implementation

    # ''' Output from EKS can be csv or DataFrame, whatever is easier for the next step '''

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
            
            # load video predictions for this particular video (for now from rng0, later from eks)
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            csv_filename = base_name + ".csv"
            preds_csv_path = os.path.join("/teamspace/studios/this_studio/outputs/mirror-mouse/100_1000-eks-random/rng0/", "video_preds", csv_filename)
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
            # append pseudo labels to hand labels
            # TODO: instead of training this 1 time. train the model 5 times based on the init_ensemble_seeds
            new_labels = pd.concat([new_labels, subselected_preds])

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

    for k in cfg["ensemble_seeds"]:
        train_and_infer(
            cfg = cfg, 
            cfg_lp= cfg_lp, 
            k = k, 
            data_dir = data_dir, 
            results_dir = results_dir, 
            min_steps=cfg["min_steps"],
            max_steps=cfg["max_steps"],
            milestone_steps=cfg["milestone_steps"],
            val_check_interval=cfg["val_check_interval"],
            video_directories=cfg["video_directories"],
            new_labels_csv=new_labels_csv
        )

        #TODO: train_and_infer() use all of the input 
            #argument: seed k, config path or actual config file (not sure if it matters), data directory, csv, all the arguemtns from pl.Train should also be pass from this function (min step, max step, eval_check)
            #resulst_dir is an argument for train_and_infer() as well
            # train_and_infer() function, min_step argument is config of min_step
            #video_directory() also need to be passed
            # Big idea: for k in seeed, run this train_and_infer(), which sit inside train.py
            # soon implement a baseline, huge long chunk of code baseline or eks, there sould be clear how we want to section that code off. 

    
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
