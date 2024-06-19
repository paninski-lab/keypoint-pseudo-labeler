"""Example pipeline script."""

#%%
import argparse
import yaml
from omegaconf import DictConfig
import sys
import os
import numpy as np
import pandas as pd
from pseudo_labeler.train import train, inference_with_metrics
from pseudo_labeler.frame_selection import select_frame_idxs_eks, export_frames
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eks')))

from eks.utils import format_data, populate_output_dataframe

# from pseudo_labeler import VIDEO_PREDS_DIR
# from eks.singleview_smoother import vectorized_ensemble_kalman_smoother_single_view
# from eks.jax_singleview_smoother import jax_ensemble_kalman_smoother_single_view

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
    
    # -------------------------------------------------------------------------------------
    # train k supervised models on n hand-labeled frames and compute labeled OOD metrics
    # -------------------------------------------------------------------------------------
    print(f'training {len(cfg["init_ensemble_seeds"])} baseline models')
    for k in cfg["init_ensemble_seeds"]:
        
        cfg_lp.data.data_dir = data_dir
        
        # update training seeds
        cfg_lp.training.rng_seed_data_pt = k
        
        # add iteration-specific fields to the config
        #translate number of steps into numbers of epoch
        cfg_lp.training.max_epochs = 10
        cfg_lp.training.min_epochs = 10
        cfg_lp.training.unfreeze_step = 30
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        results_dir = os.path.join(parent_dir, f"../outputs/mirror-mouse/100_1000-eks-random/rng{k}")
        os.makedirs(results_dir, exist_ok=True)
        
        """
        mirror-mouse/100_1000-eks-random/rng0
        mirror-mouse/100_1000-eks-random/rng1
        mirror-mouse/100_1000-eks-random/eks
        ...
        mirror-mouse/100_10000-eks-strategy2/rng0
        """

        tb_logs_dir = os.path.join(results_dir, "tb_logs/test")

        if os.path.exists(tb_logs_dir):
            print(f"tb_logs/test directory found for rng{k}. Skipping training.") 
            num_videos = 0
            for video_dir in cfg["video_directories"]:
                video_files = os.listdir(os.path.join(data_dir, video_dir))
                num_videos += len(video_files)   
            continue
        
        else:
            print(f"No tb_logs/test directory found for rng{k}. Training the model.")
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
            num_videos = 0
            for video_dir in cfg["video_directories"]:
                video_files = os.listdir(os.path.join(data_dir, video_dir))
                num_videos += len(video_files)
                for video_file in video_files:
                    results_df = inference_with_metrics(
                        video_file=os.path.join(data_dir, video_dir, video_file),
                        cfg=cfg_lp,
                        preds_file=os.path.join(results_dir, "video_preds", video_file.replace(".mp4", ".csv")),
                        ckpt_file=best_ckpt,
                        data_module=data_module,
                        trainer=trainer,
                        metrics=True,
                    )
                   
    # -------------------------------------------------------------------------------------
    # optional: run eks on all InD/OOD videos
    # # -------------------------------------------------------------------------------------
    # input_dir = vid_data  # = output from previous step
    # os.makedirs(results_dir, exist_ok=True)  # can be removed later (if results_dir exists)
    # data_type = cfg["data_type"]
    # output_df = None

    # if cfg["pseudo_labeler"] == "eks":
    #     save_filename = "EKS_output"
    #     bodypart_list = cfg_lp["keypoint_names"]
    #     s = None  # always optimize s
    #     s_frames = cfg["eks_s_frames"]  # frames used for optimizing s, all by default
    #     jax = "True"  # always use JAX acceleration for now

    #     # Load and format input files and prepare an empty DataFrame for output.
    #     input_dfs, output_df, _ = format_data(input_dir, data_type)  # check compatibility later
    #     print('Input data has been read into EKS.')

    #     ''' This region should be identical to EKS singlecam script '''
    #     # Convert list of DataFrames to a 3D NumPy array
    #     data_arrays = [df.to_numpy() for df in input_dfs]
    #     markers_3d_array = np.stack(data_arrays, axis=0)

    #     # Map keypoint names to keys in input_dfs and crop markers_3d_array
    #     keypoint_is = {}
    #     keys = []
    #     for i, col in enumerate(input_dfs[0].columns):
    #         keypoint_is[col] = i
    #     for part in bodypart_list:
    #         keys.append(keypoint_is[part + '_x'])
    #         keys.append(keypoint_is[part + '_y'])
    #         keys.append(keypoint_is[part + '_likelihood'])
    #     key_cols = np.array(keys)
    #     markers_3d_array = markers_3d_array[:, :, key_cols]

    #     # Initialize
    #     df_dicts, s_finals, nll_values = [], [], []

    #     # Call the smoother function
    #     if jax == "True":
    #         df_dicts, s_finals, nll_values_array = jax_ensemble_kalman_smoother_single_view(
    #             markers_3d_array,
    #             bodypart_list,
    #             s,
    #             s_frames
    #         )
    #     else:
    #         df_dicts, s_finals, nll_values_array = vectorized_ensemble_kalman_smoother_single_view(
    #             markers_3d_array,
    #             bodypart_list,
    #             s,
    #             s_frames
    #         )
    #     ''' end of identical region '''

    #     # Save eks results in new DataFrames and .csv output files
    #     for k in range(len(bodypart_list)):
    #         df = df_dicts[k][bodypart_list[k] + '_df']
    #         output_df = populate_output_dataframe(df, bodypart_list[k], output_df)
    #         save_filename = save_filename or f'singlecam_{s_finals[k]}.csv'
    #         output_df.to_csv(os.path.join(results_dir, save_filename))

    #     print("EKS DataFrame output successfully converted to CSV")

    # else:
    #     output_df = input_dir
    #     # other baseline pseudolaber implementation

    # ''' Output from EKS can be csv or DataFrame, whatever is easier for the next step '''


    # # -------------------------------------------------------------------------------------
    # # select frames to add to the dataset
    # # -------------------------------------------------------------------------------------
    
    print(f"Total number of videos: {num_videos}")
    
    print(
        f'selecting {cfg["n_pseudo_labels"]} pseudo-labels using {cfg["pseudo_labeler"]} '
        f'({cfg["selection_strategy"]} strategy)'
    )
    
    new_labels = pd.read_csv(os.path.join(data_dir, "CollectedData.csv"),header = [0,1,2], index_col=0)
    frames_per_video = cfg["n_pseudo_labels"] / num_videos
    print(f"Frames per video: {frames_per_video}")

    selected_frame_idxs = []    
    labeled_data_dir = os.path.join(data_dir, "labeled_data")  # Directory to save labeled frames

    for video_dir in cfg["video_directories"]:
        video_files = os.listdir(os.path.join(data_dir, video_dir))
        for video_file in video_files:         
            video_path = os.path.join(data_dir, video_dir, video_file)
            frame_idxs = select_frame_idxs_eks(
                video_file=video_file,
                n_frames_to_select=frames_per_video,
            )
            selected_frame_idxs.extend(frame_idxs)
            # debugging: print(f"Selected frame indices for {video_file}: {frame_idxs}")

            # # export frames to labeled data directory
            export_frames(
                video_file = video_path,
                save_dir = labeled_data_dir,
                frame_idxs=frame_idxs,
                format="png",
                n_digits=8,
                context_frames=0,
            )
            
    #         # load video predictions for this particular video (for now from rng0, later from eks)
            preds_df = pd.read_csv("/teamspace/studios/this_studio/outputs/mirror-mouse/100_1000-eks-random/rng0/predictions.csv",header = [0,1,2], index_col=0)
            mask = preds_df.columns.get_level_values("coords").isin(["x", "y"])
            preds_df = preds_df.loc[:, mask]

            print(preds_df.head())
            
            # subselect the predictions corresponding to frame_idxs
            subselected_preds = preds_df[preds_df.index.isin(frame_idxs)]
            
            #debugging
            print("Subselected Predictions:")
            print(subselected_preds)
            
            # append pseudo labels to hand labels
            # concatenate subselected predictions from this video to new_labels; call this new_labels also
            new_labels = pd.concat([new_labels, subselected_preds])

            #debugging
            print("New Labels:")
            print(new_labels)

    # Save the updated new_labels DataFrame 
    new_labels_file = os.path.join(data_dir, "new_labels_100_1000_eks.csv")
    new_labels.to_csv(new_labels_file, index=False)


    # # FOR LATER: check that we have the right number of labels; new_labels.shape[0] should equal cfg["n_pseudo_labels"] + 
    # # save out new_labels in a new csv file
    # # TODO

    # # - we need to add frames to the existing dataset
    # # - for each strategy/run/whatever, need to make a new csv file with updated frames

    # # train model(s) on expanded dataset
    # print(f'training {len(cfg["final_ensemble_seeds"])} baseline models')
    # for k in cfg["final_ensemble_seeds"]:
    #     # load lightning pose config file
    #     cfg_lp = use omega conf to load
    #     # - update data.data_dir, maybe some other paths
    #     # - update training.rng_seed_data_pt
    #     # - add iteration-specific fields to the config
    #     # - NEW: change labeled data field (data.csv_file)

    #     # define the output directory
    #     results_dir = todo

    #     # train model
    #     # if we run inference on videos inside train(), then we should pass a list of video
    #     # directories to loop over; these should probably be stored in pipeline config file
    #     train(cfg=cfg_lp, results_dir=results_dir)

    # # -------------------------------------------------------------------------------------
    # # run inference on all InD/OOD videos and compute unsupervised metrics
    # # -------------------------------------------------------------------------------------
    # # do as above

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

# %%
