import argparse
import os

import pandas as pd

from pseudo_labeler.evaluation import pipeline_ood_snippets
from pseudo_labeler.frame_selection import *
from pseudo_labeler.train import train_and_infer
from pseudo_labeler.utils import (
    collect_missing_eks_csv_paths,
    find_video_names,
    load_cfgs,
    pipeline_eks,
)


def pipeline(config_file: str):

    # ------
    # Setup
    # ------

    # Load cfg (pipeline yaml) and cfg_lp (lp yaml)
    cfg, cfg_lp = load_cfgs(config_file)  # cfg_lp is a DictConfig, cfg is not
    

    # Define + create directories
    data_dir = cfg_lp.data.data_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.dirname(os.path.dirname(script_dir))
    parent_dir = os.path.dirname(script_dir)
    labeled_data_dir = os.path.join(data_dir, "labeled-data")
    outputs_dir = os.path.join(parent_dir, (
        f'../outputs/{os.path.basename(data_dir)}/'
        f'hand={cfg["n_hand_labels"]}_pseudo={cfg["n_pseudo_labels"]}'
    ))
    networks_dir = os.path.join(outputs_dir, 'networks')
    pp_dir = os.path.join(
        outputs_dir,
        'post-processors',
        f"{cfg['pseudo_labeler']}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
    )
    hand_pseudo_combined = os.path.join(outputs_dir, "pseudo_label_and_hand_label")
    aeks_dir = os.path.join(
        outputs_dir,
        f"results_aeks_{cfg['selection_strategy']}"
    )

    # frame_selection_path = os.path.join(source_dir, (
    #     f"outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
    #     f"pseudo={cfg['n_pseudo_labels']}/post-quality-frame/"
    # ))

    # final_selected_frames_path = os.path.join(frame_selection_path, 'step_6_maskingNaN_file.csv')

    aeks_eks_dir = os.path.join(
        aeks_dir,
        f"{cfg['pseudo_labeler']}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
    )
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(networks_dir, exist_ok=True)
    os.makedirs(pp_dir, exist_ok=True)
    os.makedirs(hand_pseudo_combined, exist_ok=True)
    os.makedirs(aeks_dir, exist_ok=True)
    os.makedirs(aeks_eks_dir, exist_ok=True)

    # Build list of video names from the video directory
    num_videos, video_names = find_video_names(data_dir, cfg["video_directories"])
    print(f'Found {num_videos} videos: {video_names}.')

    # -------------------------------------------------------------------------------------
    # Train k supervised models on n hand-labeled frames and compute labeled OOD metrics
    # -------------------------------------------------------------------------------------

    # # Pick n hand labels. Make two csvs: one with the labels, one with the leftovers
    subsample_path, unsampled_path = pick_n_hand_labels(
        cfg.copy(), cfg_lp.copy(), data_dir, outputs_dir)

    # ||| Main first-round training loop |||
    # loops over ensemble seeds training a model for each seed with n hand_labels
    print(f'Training {len(cfg["ensemble_seeds"])} baseline models.')
    for k in cfg["ensemble_seeds"]:
        # Make directory for rng{seed}
        results_dir = os.path.join(networks_dir, f'rng{k}')
        os.makedirs(results_dir, exist_ok=True)
        # Main function call
        train_and_infer(
            cfg=cfg.copy(),
            cfg_lp=cfg_lp.copy(),
            k=k,
            data_dir=data_dir,
            results_dir=results_dir,
            new_labels_csv=subsample_path  # Set to None to use the original csv_file
        )

    # # # -------------------------------------------------------------------------------------
    # # # Post-process network outputs to generate potential pseudo labels (chosen in next step)
    # # # -------------------------------------------------------------------------------------
    pseudo_labeler = cfg["pseudo_labeler"]
    # Collect input eks csv paths from video names; skip existing
    eks_csv_paths = collect_missing_eks_csv_paths(video_names, pp_dir)
    print(f'Post-processing the following videos using {pseudo_labeler}: {eks_csv_paths}')
    # ||| Main EKS function call ||| pipeline_eks will also handle ensemble_mean baseline
    if pseudo_labeler == "eks" or pseudo_labeler == "ensemble_mean":
        pipeline_eks(
            input_csv_names=eks_csv_paths,
            input_dir=networks_dir,
            data_type=cfg["data_type"],
            pseudo_labeler=pseudo_labeler,
            cfg_lp=cfg_lp.copy(),
            results_dir=pp_dir
        )

    # # -------------------------------------------------------------------------------------
    # # run inference on OOD snippets (if specified) -- using network models
    # # -------------------------------------------------------------------------------------
    dataset_name = os.path.basename(data_dir)
    if cfg["ood_snippets"]:
        print(f'Starting OOD snippet analysis for {dataset_name}')
        pipeline_ood_snippets(
            cfg=cfg,
            cfg_lp=cfg_lp,
            data_dir=data_dir,
            networks_dir=networks_dir,
            pp_dir=pp_dir,
            pseudo_labeler=pseudo_labeler
        )

    # # -------------------------------------------------------------------------------------
    # # select frames to add to the dataset
    # # -------------------------------------------------------------------------------------
    selection_strategy = cfg["selection_strategy"]
    print(
        f'Selecting {cfg["n_pseudo_labels"]} pseudo-labels from {num_videos} '
        f'{cfg["pseudo_labeler"]} outputs using ({selection_strategy} strategy)'
    )
    hand_labels = pd.read_csv(subsample_path, header=[0, 1, 2], index_col=0)
    # Process each ensemble seed
    for k in cfg["ensemble_seeds"]:
        # Initialize seed_labels with hand labels for this seed
        seed_labels = hand_labels.copy()
        combined_csv_filename = (
            f"CollectedData_hand={cfg['n_hand_labels']}"
            f"_pseudo={cfg['n_pseudo_labels']}_k={k}_{selection_strategy}.csv"
        )
        combined_csv_path = os.path.join(hand_pseudo_combined, combined_csv_filename)

        # Check if frame selection has already been done
        if os.path.exists(combined_csv_path):
            print(
                f'Selected frames already exist at {combined_csv_path}. '
                f'Skipping frame selection for rng{k}.'
            )
            seed_labels = pd.read_csv(combined_csv_path, header=[0, 1, 2], index_col=0)

        else:
            print(f'Selecting pseudo-labels using a {selection_strategy} strategy.')

            if selection_strategy == 'random':
                seed_labels = select_frames_random(
                    cfg=cfg.copy(),
                    k=k,
                    data_dir=data_dir,
                    num_videos=num_videos,
                    pp_dir=pp_dir,
                    labeled_data_dir=labeled_data_dir,
                    seed_labels=seed_labels
                )

            elif selection_strategy == 'hand':
                seed_labels = select_frames_hand(
                    unsampled_path=unsampled_path,
                    n_frames_to_select=cfg['n_pseudo_labels'],
                    k=k,
                    seed_labels=seed_labels
                )
                
            elif selection_strategy == 'frame_selection':
                frame_selection_path = os.path.join(source_dir, (
                    f"outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
                    f"pseudo={cfg['n_pseudo_labels']}/post-quality-frame/"
                ))

                final_selected_frames_path = select_frames_strategy_pipeline(
                    cfg=cfg.copy(),
                    cfg_lp=cfg_lp.copy(),
                    data_dir=data_dir,
                    source_dir=source_dir,
                    frame_selection_path=frame_selection_path
                )

                seed_labels = process_and_export_frame_selection(
                    cfg=cfg.copy(),
                    cfg_lp=cfg_lp.copy(),
                    data_dir=data_dir,
                    labeled_data_dir=labeled_data_dir,
                    final_selected_frames_path=final_selected_frames_path,
                    seed_labels=seed_labels
                )
            
            # Export the combined hand labels and pseudo labels for this seed
            seed_labels.to_csv(combined_csv_path)
            print(
                f"Saved combined hand labels and pseudo labels for seed {k} to "
                f"{combined_csv_path}"
            )
            
        # Check number of labels for this seed
        expected_total_labels = cfg['n_hand_labels'] + cfg["n_pseudo_labels"]
        if seed_labels.shape[0] != expected_total_labels:
            print(
                f"Warning: Number of labels for seed {k} ({seed_labels.shape[0]}) "
                f"does not match expected count ({expected_total_labels})"
            )
        else:
            print(f"Label count verified for seed {k}: {seed_labels.shape[0]} labels")

        # # -------------------------------------------------------------------------------------
        # # Train models on expanded dataset
        # # -------------------------------------------------------------------------------------

        csv_prefix = (
            f"hand={cfg['n_hand_labels']}_rng={k}_"
            f"pseudo={cfg['n_pseudo_labels']}_"
            f"{cfg['pseudo_labeler']}_{cfg['selection_strategy']}_"
            f"rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
        )
        results_dir = os.path.join(aeks_dir, f"rng{k}")

        # Run train_and_infer with the combined hand labels and pseudo labels
        train_and_infer(
            cfg=cfg.copy(),
            cfg_lp=cfg_lp.copy(),
            k=k,
            data_dir=data_dir,
            results_dir=results_dir,
            csv_prefix=csv_prefix,
            new_labels_csv=combined_csv_path,  # Use the combined CSV file for this seed
            n_train_frames=expected_total_labels
        )

        print(
            f"Completed training and inference for seed {k} "
            f"using combined hand labels and pseudo labels"
        )

    print("Completed training and inference for all seeds using expanded datasets")

    # # # -------------------------------------------------------------------------------------
    # # # Run EKS on expanded dataset inferences
    # # # -------------------------------------------------------------------------------------
    pseudo_labeler = 'eks'
    # Collect input csv names from video names; skip existing ones
    eks_csv_paths = collect_missing_eks_csv_paths(video_names, aeks_eks_dir)
    print(f'Post-processing the following videos using {pseudo_labeler}: {eks_csv_paths}')
    # ||| Main second round EKS function call |||
    pipeline_eks(
        input_csv_names=eks_csv_paths,
        input_dir=aeks_dir,
        data_type=cfg["data_type"],
        pseudo_labeler=pseudo_labeler,
        cfg_lp=cfg_lp.copy(),
        results_dir=results_dir
    )

    # # -------------------------------------------------------------------------------------
    # # run inference on OOD snippets (if specified) -- using network models
    # # -------------------------------------------------------------------------------------
    dataset_name = os.path.basename(data_dir)
    if cfg["ood_snippets"]:
        print(f'Starting OOD snippet analysis for {dataset_name}')
        pipeline_ood_snippets(
            cfg=cfg,
            cfg_lp=cfg_lp,
            data_dir=data_dir,
            networks_dir=aeks_dir,
            pp_dir=aeks_eks_dir,
            pseudo_labeler=pseudo_labeler
        )
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='absolute path to .yaml configuration file',
        type=str,
    )
    args = parser.parse_args()
    pipeline(args.config)
