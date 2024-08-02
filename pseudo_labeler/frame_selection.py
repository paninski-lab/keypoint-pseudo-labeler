"""Function to select pseudo-labeled frames."""

import os
from typing import List, Dict, Any
import cv2
import numpy as np
import pandas as pd
from pseudo_labeler.video import get_frames_from_idxs


def get_total_frames(video_file: str) -> int:
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'{total_frames} frames detected in {os.path.basename(video_file)}')
    
    # Release the video capture object
    cap.release()
    
    return total_frames

# --------------------------
# Frame selection functions:
# --------------------------
def select_frames_random(
    cfg: Dict[str, Any], k: int, data_dir: str, num_videos: int, pp_dir: str, labeled_data_dir: str, seed_labels: Dict[str, Any]
) -> Dict[str, Any]:
    video_directories = cfg["video_directories"]
    n_pseudo_labels = cfg["n_pseudo_labels"]
    
    frames_per_video = int(n_pseudo_labels / num_videos)
    print(f"Frames per video: {frames_per_video}")

    for video_dir in video_directories:
        video_files = os.listdir(os.path.join(data_dir, video_dir))
        for video_file in video_files:
            video_path = os.path.join(data_dir, video_dir, video_file)
            
            # Get total number of frames in the video
            total_frames = get_total_frames(video_path)
            
            # Select random frame indices
            np.random.seed(k)
            frame_idxs = np.random.choice(total_frames, frames_per_video, replace=False)
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_filename = base_name + ".csv"
            preds_csv_path = os.path.join(pp_dir, csv_filename)
            frame_idxs = frame_idxs.astype(int)
            print(f'Selected frame indices (displaying first 10 of {len(frame_idxs)}): {frame_idxs[0:10]}...')
            
            export_frames(
                video_file=video_path,
                save_dir=os.path.join(labeled_data_dir, base_name),
                frame_idxs=frame_idxs,
                format="png",
                n_digits=8,
                context_frames=0,
            )
            
            preds_df = pd.read_csv(preds_csv_path, header=[0, 1, 2], index_col=0)
            subselected_preds = process_predictions(preds_df, frame_idxs, base_name, generate_index=True)
            
            seed_labels = update_seed_labels(seed_labels, subselected_preds)
    
    return seed_labels


def select_frames_hand(
    unsampled_path: str, n_frames_to_select: int, k: int,
    seed_labels: Dict[str, Any]
) -> Dict[str, Any]:
    # Read the CSV file, skipping the first three rows
    df = pd.read_csv(unsampled_path, skiprows=2)
    
    # Get the total number of rows (frames)
    total_rows = df.shape[0]
    
    # If there are not enough rows, return all of the rows
    if total_rows <= n_frames_to_select:
        frame_idxs = np.arange(total_rows)
    else:
        # Set the random seed for reproducibility
        np.random.seed(k)
        # Randomly select n_frames_to_select rows
        frame_idxs = np.random.choice(total_rows, n_frames_to_select, replace=False)
    
    frame_idxs = frame_idxs.astype(int)
    print(f'Selected frame indices (displaying first 10 of {len(frame_idxs)}): {frame_idxs[0:10]}...')
    
    preds_csv_path = unsampled_path
    preds_df = pd.read_csv(preds_csv_path, header=[0, 1, 2], index_col=0)
    base_name = os.path.splitext(os.path.basename(preds_csv_path))[0]
    subselected_preds = process_predictions(preds_df, frame_idxs, base_name, generate_index=False)
    
    updated_seed_labels = update_seed_labels(seed_labels, subselected_preds)
    return updated_seed_labels


def export_frames(
    video_file: str,
    save_dir: str,
    frame_idxs: np.ndarray,
    format: str = "png",
    n_digits: int = 8,
    context_frames: int = 0,
) -> None:
    """

    Parameters
    ----------
    video_file: absolute path to video file from which to select frames
    save_dir: absolute path to directory in which selected frames are saved
    frame_idxs: indices of frames to grab
    format: only "png" currently supported
    n_digits: number of digits in image names
    context_frames: number of frames on either side of selected frame to also save

    """

    cap = cv2.VideoCapture(video_file)

    # expand frame_idxs to include context frames
    if context_frames > 0:
        context_vec = np.arange(-context_frames, context_frames + 1)
        frame_idxs = (frame_idxs[None, :] + context_vec[:, None]).flatten()
        frame_idxs.sort()
        frame_idxs = frame_idxs[frame_idxs >= 0]
        frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_idxs = np.unique(frame_idxs)
    

    # load frames from video
    frames = get_frames_from_idxs(cap, frame_idxs)

    # save out frames
    os.makedirs(save_dir, exist_ok=True)
    for frame, idx in zip(frames, frame_idxs):
        cv2.imwrite(
            filename=os.path.join(save_dir, "img%s.%s" % (str(idx).zfill(n_digits), format)),
            img=frame[0],
        )


def pick_n_hand_labels(cfg, cfg_lp, data_dir, outputs_dir):
    """
    Subsamples n hand labels from the dataset and saves them along with the remaining labels.

    Args:
    - cfg (dict): Configuration dictionary containing 'pipeline_seeds'.
    - cfg_lp (dict): Configuration dictionary containing 'training.train_frames'.
    - data_dir (str): Directory containing the full dataset.
    - outputs_dir (str): Directory to save the subsample and unsampled files.

    Returns:
    - subsampled_path: Path to the subsampled csv containing picked hand labels.
    """

    # Set pipeline seed -- this is used for selecting n hand labels for training networks
    np.random.seed(cfg["pipeline_seeds"])

    # Create subsample file (csv with selected hand labels)
    subsample_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_p={cfg['pipeline_seeds']}.csv"
    subsample_path = os.path.join(outputs_dir, subsample_filename)

    # Create unsampled file (csv of the leftover rows after sampling hand labels)
    unsampled_filename = f"CollectedData_hand={cfg_lp.training.train_frames}_p={cfg['pipeline_seeds']}_unsampled.csv"
    unsampled_path = os.path.join(outputs_dir, unsampled_filename)

    # Load the full dataset
    collected_data = pd.read_csv(os.path.join(data_dir, "CollectedData.csv"), header=[0,1,2])

    # Save hand labels in the subsample csv
    initial_subsample = collected_data.sample(n=cfg_lp.training.train_frames)
    initial_subsample.to_csv(subsample_path, index=False)
    print(f"Saved initial subsample hand labels CSV file: {subsample_path}")

    # Save remaining rows into the unsampled csv
    initial_indices = initial_subsample.index
    unsampled = collected_data.drop(index=initial_indices)
    unsampled.to_csv(unsampled_path, index=False)
    print(f"Saved unsampled hand labels CSV file: {unsampled_path}")
    return subsample_path, unsampled_path


def process_predictions(preds_df, frame_idxs, base_name, standard_scorer_name='standard_scorer', generate_index=False):
    mask = preds_df.columns.get_level_values("coords").isin(["x", "y"])
    preds_df = preds_df.loc[:, mask]
    subselected_preds = preds_df.iloc[frame_idxs]

    if generate_index:
        def generate_new_index(idx, base_name):
            return f"labeled-data/{base_name}/img{str(idx).zfill(8)}.png"

        new_index = [generate_new_index(idx, base_name) for idx in subselected_preds.index]
        subselected_preds.index = new_index

    new_columns = pd.MultiIndex.from_arrays([
        [standard_scorer_name] * len(subselected_preds.columns),
        subselected_preds.columns.get_level_values('bodyparts'),
        subselected_preds.columns.get_level_values('coords')
    ], names=['scorer', 'bodyparts', 'coords'])
    
    subselected_preds.columns = new_columns
    return subselected_preds

def update_seed_labels(seed_labels, new_preds, standard_scorer_name='standard_scorer'):
    if not seed_labels.empty:
        seed_labels.columns = pd.MultiIndex.from_arrays([
            [standard_scorer_name] * len(seed_labels.columns),
            seed_labels.columns.get_level_values('bodyparts'),
            seed_labels.columns.get_level_values('coords')
        ], names=['scorer', 'bodyparts', 'coords'])
    seed_labels = pd.concat([seed_labels, new_preds])
    return seed_labels