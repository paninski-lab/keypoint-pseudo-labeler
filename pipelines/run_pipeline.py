"""Example pipeline script."""

#%%
import argparse
import yaml
from omegaconf import DictConfig
import sys
import os
import numpy as np
from utils import format_data_walk
from pseudo_labeler.train import train, inference_with_metrics


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../eks')))

from eks.utils import populate_output_dataframe
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam

#%%
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
        # update version that works
        # cfg_lp.training.max_steps = 64
        # cfg_lp.training.min_steps = 64
        cfg_lp.training.unfreeze_step = 30
        
        # define the output directory - the name below should come from (fully generated from) config file configuration
        #result_dir should be an absolute path
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

        # train model
        # if we run inference on videos inside train(), then we should pass a list of video
        # directories to loop over; these should probably be stored in pipeline config file
        # train(cfg=cfg_lp, results_dir=results_dir)

        #TODO: Tommy - make more argument to train, max step min step, milestone, etc. pass
        #information from pipeline config which we call cfg. 
        best_ckpt, data_module, trainer = train(cfg=cfg_lp, results_dir=results_dir)
    
    # -------------------------------------------------------------------------------------
    # run inference on all InD/OOD videos and compute unsupervised metrics
    # -------------------------------------------------------------------------------------
        # this is actually already in the train function - do we want to split it?
        # iterate through all the videos in the video_dir in pipeline_example.yaml
        video_names = []
        for video_dir in cfg["video_directories"]:
            video_files = os.listdir(os.path.join(data_dir, video_dir))
            for video_file in video_files:
                video_name = video_file.replace(".mp4", ".csv")
                video_names.append(video_name)
                results_df = inference_with_metrics(
                    video_file=os.path.join(data_dir, video_dir, video_file),
                    cfg=cfg_lp,
                    preds_file=os.path.join(results_dir, "video_preds", video_name),
                    ckpt_file=best_ckpt,
                    # model=model,
                    # ckpt_file=os.path.join(results_dir, "model.ckpt"),
                    data_module=data_module,
                    trainer=trainer,
                    metrics=True,
                )


    # -------------------------------------------------------------------------------------
    # optional: run eks on all InD/OOD videos
    # # -------------------------------------------------------------------------------------
    print("Starting EKS")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, f"../outputs/mirror-mouse/100_1000-eks-random")
    input_dir = results_dir
    os.makedirs(results_dir, exist_ok=True)  # can be removed later (if results_dir exists)
    eks_dir = os.path.join(results_dir, "eks")
    os.makedirs(eks_dir, exist_ok=True)  # Ensure eks directory exists
    data_type = cfg["data_type"]
    output_df = None

    if cfg["pseudo_labeler"] == "eks":
        bodypart_list = cfg_lp["data"]["keypoint_names"]
        s = None  # optimize s
        s_frames = [(None, None)] # use all frames for optimization
        
        for video_name in video_names:
            # Load and format input files and prepare an empty DataFrame for output.
            input_dfs, output_df, _ = format_data_walk(input_dir, data_type, video_name)
            
            print(f'Input data for {video_name} has been read into EKS.')

            ''' This region should be identical to EKS singlecam script '''
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
            df_dicts, s_finals, nll_values_array = ensemble_kalman_smoother_singlecam(
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
                csv_filename = video_name
                output_path = os.path.join(eks_dir, csv_filename)
                output_df.to_csv(output_path)

            print(f"EKS DataFrame output for {video_name} successfully converted to CSV. See at {output_path}")

        else:
            output_df = input_dir
            # other baseline pseudolaber implementation



    # # -------------------------------------------------------------------------------------
    # # select frames to add to the dataset
    # # -------------------------------------------------------------------------------------
    # # TODO: 
    # print(
    #     f'selecting {cfg["n_pseudo_labels"]} pseudo-labels using {cfg["pseudo_labeler"]} '
    #     f'({cfg["selection_strategy"]} strategy)'
    # )
    # # load hand labels csv file as a pandas dataframe
    # # TODO: load hand labels, ie. CollectedData.csv; call this new_labels
    # # loop through videos and select labels from each
    # frames_per_video = cfg["n_pseudo_labels"] / num_videos  # TODO: define num_videos above
    # for video_dir in cfg["video_directories"]:
    #     video_files = os.listdir(os.path.join(data_dir, video_dir))
    #     for video_file in video_files:
    #         # select labels from this video
    #         frame_idxs = select_frame_idxs_eks(
    #             video_file=None,
    #             n_frames_to_select=frames_per_video,
    #         )
    #         # export frames to labeled data directory
    #         export_frames(
    #             video_file: str,
    #             save_dir: str,
    #             frame_idxs=frame_idxs,
    #             format="png",
    #             n_digits=8,
    #             context_frames=0,
    #         )
    #         # append pseudo labels to hand labels
    #         # TODO: load predictions for the specific indices returned by select_frame_idxs_eks()
    #         # load video predictions for this particular video (for now from rng0, later from eks)
    #         # subselect the predictions corresponding to frame_idxs
    #         # concatenate subselected predictions from this video to new_labels; call this new_labels also

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
