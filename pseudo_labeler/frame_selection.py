"""Function to select pseudo-labeled frames."""

import os
import re
import csv
from math import ceil
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


# ------------------------------------------
# Frame selection functions - random & hand
# ------------------------------------------
def select_frames_random(cfg: Dict[str, Any],
                         k: int,
                         data_dir: str,
                         num_videos: int,
                         pp_dir: str,
                         labeled_data_dir: str,
                         seed_labels: Dict[str, Any]) -> Dict[str, Any]:
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
            print(f'Selected frame indices (displaying first 10 of {len(frame_idxs)}):'
                  f' {frame_idxs[0:10]}...')

            export_frames(
                video_file=video_path,
                save_dir=os.path.join(labeled_data_dir, base_name),
                frame_idxs=frame_idxs,
                format="png",
                n_digits=8,
                context_frames=0,
            )

            preds_df = pd.read_csv(preds_csv_path, header=[0, 1, 2], index_col=0)
            subselected_preds = process_predictions(
                preds_df, frame_idxs, base_name, generate_index=True)

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
    print(f'Selected frame indices (displaying first 10 of {len(frame_idxs)}):'
          f' {frame_idxs[0:10]}...')

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
    subsample_filename = (f"CollectedData_hand={cfg_lp.training.train_frames}_"
                          f"p={cfg['pipeline_seeds']}.csv")
    subsample_path = os.path.join(outputs_dir, subsample_filename)

    # Create unsampled file (csv of the leftover rows after sampling hand labels)
    unsampled_filename = (f"CollectedData_hand={cfg_lp.training.train_frames}_"
                          f"p={cfg['pipeline_seeds']}_unsampled.csv")
    unsampled_path = os.path.join(outputs_dir, unsampled_filename)

    # Load the full dataset
    collected_data = pd.read_csv(os.path.join(data_dir, "CollectedData.csv"), header=[0, 1, 2])

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


def process_predictions(
    preds_df: pd.DataFrame,
    frame_idxs: list,
    base_name: str,
    standard_scorer_name: str = 'standard_scorer',
    generate_index: bool = False
) -> pd.DataFrame:
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



# ------------------------------------------
# Frame selection strategy - frame selection strategy with std < threshold
# ------------------------------------------

def extract_video_names(video_dir_path):
    """Extract video names from a directory."""
    video_names = set()
    if os.path.exists(video_dir_path):
        video_files = os.listdir(video_dir_path)
        for video_file in video_files:
            if video_file.lower().endswith(('.mp4')):
                video_name = os.path.splitext(video_file)[0]
                video_names.add(video_name)
    return video_names


def find_keypoints_below_ensemble_std(
    video_name: str,
    file_paths_template: str,
    output_file_path: str,
    n_ensembles: int,
    pixel_threshold: Union[int, float] = 5
) -> pd.DataFrame:
    """Step 1: Find and filter keypoints below ensemble standard deviation threshold."""
    dfs: List[pd.DataFrame] = []
    
    for idx in range(n_ensembles):
        file_path = file_paths_template.format(
            rng_idx=idx, video_name=video_name)
        print(f"Processing file: {file_path}")
        
        data = pd.read_csv(file_path)
        body_parts = data.iloc[0, 1::3].tolist()
        
        # Prepare the data
        image_paths = data.iloc[2:, 0].apply(lambda x: f"{video_name}/{x}")
        coords = data.iloc[2:, 1:].values.reshape(-1, len(body_parts), 3)
        
        # Create DataFrame for this ensemble
        df = pd.DataFrame({
            'image_path': np.repeat(image_paths, len(body_parts)),
            'body_part': body_parts * len(image_paths),
            f'x_rng{idx}': coords[:, :, 0].flatten(),
            f'y_rng{idx}': coords[:, :, 1].flatten(),
            f'likelihood_rng{idx}': coords[:, :, 2].flatten()
        })
        
        # Convert coordinate columns to numeric
        cols_to_convert = [f'x_rng{idx}', f'y_rng{idx}', f'likelihood_rng{idx}']
        df[cols_to_convert] = df[cols_to_convert].apply(
            pd.to_numeric, errors='coerce'
        )
        dfs.append(df)
    
    # Merge all DataFrames
    merged_data = dfs[0]
    for df in dfs[1:]:
        merged_data = pd.merge(merged_data, df, on=['image_path', 'body_part'])
    
    # Calculate ensemble statistics
    x_cols = [f'x_rng{i}' for i in range(n_ensembles)]
    y_cols = [f'y_rng{i}' for i in range(n_ensembles)]
    
    merged_data['ensemble_mean_x'] = merged_data[x_cols].mean(axis=1)
    merged_data['ensemble_mean_y'] = merged_data[y_cols].mean(axis=1)
    merged_data['ensemble_median_x'] = merged_data[x_cols].median(axis=1)
    merged_data['ensemble_median_y'] = merged_data[y_cols].median(axis=1)
    merged_data['ensemble_variance_x'] = merged_data[x_cols].var(axis=1)
    merged_data['ensemble_variance_y'] = merged_data[y_cols].var(axis=1)
    
    merged_data['ensemble_std'] = np.sqrt(
        merged_data['ensemble_variance_x'] + merged_data['ensemble_variance_y']
    )
    
    # Output check for total number of keypoints before filtering
    total_keypoints_before = len(merged_data)
    print(f"Total keypoints before filtering: {total_keypoints_before}")
    
    # Filter valid keypoints
    valid_keypoints = merged_data[
        merged_data['ensemble_std'] < pixel_threshold
    ]
    
    # Output check for total number of keypoints after filtering
    total_keypoints_after = len(valid_keypoints)
    print(
        f"Total keypoints after filtering "
        f"(ensemble_std < {pixel_threshold}): {total_keypoints_after}"
    )
    print(
        f"Percentage of keypoints retained: "
        f"{(total_keypoints_after / total_keypoints_before) * 100:.2f}%"
    )
    
    # Save to CSV with 'step_1_' prefix
    output_file = os.path.join(
        output_file_path, f"step_1_{video_name}_keypoints.csv"
    )
    valid_keypoints.to_csv(output_file, index=False)
    print(f"Valid keypoints data for {video_name} saved to {output_file}")

    return valid_keypoints


def find_frames_with_valid_keypoints(
    input_csv_path: str,
    output_dir: str,
    video_name: str,
    total_keypoints: int = 14,
    pixel_threshold: float = 5,
    required_percentage: float = 0.75
) -> List[str]:
    """Step 2: Find frames with valid keypoints based on ensemble standard deviation."""
    required_keypoints = ceil(required_percentage * total_keypoints)

    df = pd.read_csv(input_csv_path)

    def process_frame(group: pd.DataFrame) -> Tuple[int, bool]:
        valid_keypoints = sum(group['ensemble_std'] < pixel_threshold)
        return valid_keypoints, valid_keypoints >= required_keypoints

    result = df.groupby('image_path').apply(process_frame)

    valid_frames = [
        frame for frame, (count, is_valid) in result.items() if is_valid
    ]
    frame_counts: Dict[str, int] = {
        frame: count for frame, (count, is_valid) in result.items() if is_valid
    }

    total_frames = len(result)
    valid_frames_count = len(valid_frames)
    percentage_valid_frames = (valid_frames_count / total_frames) * 100

    print(f"Video: {video_name}")
    print(f"Total number of frames: {total_frames}")
    print(
        f"Number of frames with at least {required_keypoints} keypoints "
        f"having ensemble std dev < {pixel_threshold} pixels: "
        f"{valid_frames_count}"
    )
    print(
        f"Percentage of frames with at least {required_keypoints} keypoints "
        f"having ensemble std dev < {pixel_threshold} pixels: "
        f"{percentage_valid_frames:.2f}%"
    )

    output_path = os.path.join(
        output_dir, f'step_2_{video_name}_valid_frames.txt'
    )
    with open(output_path, 'w') as f:
        for frame, count in frame_counts.items():
            f.write(f"{frame}: {count}\n")

    print(f"Valid frames with counts have been saved to {output_path}")

    output_csv_path = os.path.join(
        output_dir, f'step_2_{video_name}_valid_frames.csv'
    )
    pd.DataFrame(valid_frames, columns=['frame']).to_csv(
        output_csv_path, index=False, header=False
    )
    print(f"Valid frames have been saved to {output_csv_path}")

    return valid_frames


def eks_output_low_stdev_frames(
    video_name: str,
    eks_file_path: str,
    valid_frames_path: str,
    output_dir: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Step 3-1: Filter EKS output for frames with low standard deviation."""
    with open(eks_file_path, 'r') as f:
        header_rows = [f.readline().strip() for _ in range(2)]
    
    eks_df = pd.read_csv(eks_file_path, skiprows=2)
    
    eks_df.iloc[:, 0] = f"{video_name}/" + eks_df.iloc[:, 0].astype(str)
    
    valid_frames = pd.read_csv(valid_frames_path, header=None, names=['frame'])
    filtered_eks = pd.merge(eks_df, valid_frames, left_on=eks_df.columns[0], right_on='frame', how='inner')

    missing_frames = set(valid_frames['frame']) - set(filtered_eks.iloc[:, 0])
    
    if missing_frames:
        print(f"\nFrames in step 2 that are missing from filtered EKS for {video_name}:")
        print(*missing_frames, sep='\n')
    else:
        print(f"\nAll frames from step 2 are present in filtered EKS for {video_name}.")

    output_file = os.path.join(output_dir, f'step_3_{video_name}_eks_output_low_stdev_frames.csv')
    with open(output_file, 'w') as f:
        f.writelines([header + '\n' for header in header_rows])
        filtered_eks.to_csv(f, index=False)
    
    print(f"Filtered EKS data for {video_name} saved to {output_file}")
    print(f"Number of frames in filtered data: {len(filtered_eks)}")

    return filtered_eks, header_rows


def format_filtered_eks_data(
    all_data_frames: List[pd.DataFrame],
    header_rows: List[str],
    output_dir: str
) -> pd.DataFrame:
    """Step 3-2: Format and combine filtered EKS data from multiple videos."""
    all_eks_output = pd.concat(all_data_frames, ignore_index=True)
    
    def drop_columns(
        df: pd.DataFrame,
        headers: List[str],
        substrings: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        cols_to_drop = [
            col for col in df.columns if any(sub in col for sub in substrings)
        ]
        cols_to_drop_indices = [df.columns.get_loc(col) for col in cols_to_drop]
        
        df = df.drop(columns=cols_to_drop)
        
        new_headers = []
        for header in headers:
            header_parts = header.split(',')
            new_header_parts = [
                part for i, part in enumerate(header_parts)
                if i not in cols_to_drop_indices
            ]
            new_headers.append(','.join(new_header_parts))
        
        return df, new_headers
    
    substrings_to_drop = ['zscore', 'likelihood']
    all_eks_output, header_rows = drop_columns(
        all_eks_output, header_rows, substrings_to_drop
    )
    
    all_eks_output = all_eks_output.drop(columns=['frame'])
    
    new_columns = [
        'x' if 'x.' in col or col == 'x' else
        'y' if 'y.' in col or col == 'y' else col
        for col in all_eks_output.columns
    ]
    
    all_eks_output.columns = new_columns
    
    all_output_file = os.path.join(
        output_dir, 'step_4_all_eks_output_low_stdev_frames.csv'
    )
    with open(all_output_file, 'w') as f:
        f.writelines([header + '\n' for header in header_rows])
        all_eks_output.to_csv(f, index=False)
    
    print(f"\nAll filtered EKS data saved to {all_output_file}")
    print(f"Total number of frames in combined data: {len(all_eks_output)}")
    
    return all_eks_output


def run_kmeans_on_eks_output(
    eks_data_path: str,
    n_clusters: int = 1000
) -> Tuple[pd.DataFrame, List[str]]:
    """Step 4: Perform K-means clustering on EKS output data."""
    with open(eks_data_path, 'r') as f:
        header = [f.readline().strip() for _ in range(2)]

    eks_data = pd.read_csv(eks_data_path, skiprows=2)

    x_coords = eks_data.filter(regex='^x').values
    y_coords = eks_data.filter(regex='^y').values

    # Stack x and y coordinates
    coords = np.hstack((x_coords, y_coords))

    # Determine the number of components for PCA
    n_components = min(coords.shape[0], coords.shape[1], 32)

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(coords)

    n_clusters = min(n_clusters, embedding.shape[0])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding)

    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(embedding[:, None, :] - centers[None, :, :], axis=2)
    prototype_indices = np.argmin(distances, axis=0)
    prototype_frames = eks_data.iloc[prototype_indices]

    return prototype_frames, header


def update_coordinates(valid_keypoints_path: str, clustered_frames_path: str) -> List[List[str]]:
    """Update coordinates based on valid keypoints and KNN clustered frames data."""
    valid_keypoints = set()
    with open(valid_keypoints_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            valid_keypoints.add((row[0], row[1]))  # (image_path, body_part)

    with open(clustered_frames_path, 'r') as f:
        clustered_data = list(csv.reader(f))

    headers = clustered_data[:2]
    body_parts = headers[1][1::2]

    coords_row = ['coords']
    for col in clustered_data[2][1:]:
        coords_row.append(col[0] if col.startswith(('x', 'y')) else col)

    updated_headers = headers + [coords_row]
    updated_data = updated_headers.copy()

    for row in clustered_data[3:]:
        frame = row[0]
        new_row = [frame]
        for i, body_part in enumerate(body_parts):
            if (frame, body_part) in valid_keypoints:
                new_row.extend([row[2*i+1], row[2*i+2]])
            else:
                new_row.extend(['NaN', 'NaN'])
        updated_data.append(new_row)

    return updated_data


def select_frames_strategy_pipeline(
    cfg: dict,
    cfg_lp: dict,
    data_dir: str,
    source_dir: str,
    frame_selection_path: str
) -> str:
    """Run the frame selection pipeline and return the path of the final selected frames."""
    final_selected_frames_path = os.path.join(frame_selection_path, 'step_6_maskingNaN_file.csv')

    if os.path.exists(final_selected_frames_path):
        print(f"\n\nFinal output file {final_selected_frames_path} already exists. "
              "Skipping the entire frame selection process.\n")
        return final_selected_frames_path

    os.makedirs(frame_selection_path, exist_ok=True)

    video_names = set()
    for video_dir in cfg["video_directories"]:
        video_dir_path = os.path.join(data_dir, video_dir)
        video_names.update(extract_video_names(video_dir_path))
    video_names = list(video_names)

    experiment_dir = (f"outputs/{os.path.basename(data_dir)}/"
                      f"hand={cfg_lp.training.train_frames}_"
                      f"pseudo={cfg['n_pseudo_labels']}")
    file_paths_template = os.path.join(source_dir, experiment_dir, 'networks',
                                       'rng{rng_idx}', 'video_preds', '{video_name}.csv')

    all_keypoints = []
    for video_name in video_names:
        keypoints = find_keypoints_below_ensemble_std(
            video_name=video_name,
            file_paths_template=file_paths_template,
            output_file_path=frame_selection_path,
            n_ensembles=len(cfg["ensemble_seeds"]),
            pixel_threshold=cfg["pixel_threshold"] #5
        )
        all_keypoints.append(keypoints)

    combined_keypoints = pd.concat(all_keypoints, ignore_index=True)
    all_videos_valid_keypoints_path = os.path.join(frame_selection_path, "step_1_all_videos_keypoints.csv")
    combined_keypoints.to_csv(all_videos_valid_keypoints_path, index=False)

    for video_name in video_names:
        input_csv_path = os.path.join(frame_selection_path, f"step_1_{video_name}_keypoints.csv")
        find_frames_with_valid_keypoints(
            input_csv_path=input_csv_path,
            output_dir=frame_selection_path,
            video_name=video_name,
            total_keypoints=cfg_lp.data.num_keypoints,
            pixel_threshold= cfg["pixel_threshold"], #5,
            required_percentage= cfg["required_keypoint_percentage"] #0.75
        )

    all_filtered_data_frames = []
    header_rows = None

    for video_name in video_names:
        eks_file_path = os.path.join(
            source_dir, 
            f"outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
            f"pseudo={cfg['n_pseudo_labels']}/post-processors/"
            f"{cfg['pseudo_labeler']}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}",
            f"{video_name}.csv"
        )
        valid_frames_path = os.path.join(frame_selection_path, f'step_2_{video_name}_valid_frames.csv')
        
        filtered_data, file_header_rows = eks_output_low_stdev_frames(
            video_name=video_name,
            eks_file_path=eks_file_path,
            valid_frames_path=valid_frames_path,
            output_dir=frame_selection_path
        )
        
        all_filtered_data_frames.append(filtered_data)
        
        if header_rows is None:
            header_rows = file_header_rows

    format_filtered_eks_data(
        all_data_frames=all_filtered_data_frames,
        header_rows=header_rows,
        output_dir=frame_selection_path
    )

    eks_data_path = os.path.join(frame_selection_path, 'step_4_all_eks_output_low_stdev_frames.csv')
    prototype_frames, header = run_kmeans_on_eks_output(eks_data_path, n_clusters=1000)

    kmeans_frames_path = os.path.join(frame_selection_path, 'step_5_kmeans_prototype_frames.csv')
    with open(kmeans_frames_path, 'w') as f:
        f.write('\n'.join(header) + '\n')
        prototype_frames.to_csv(f, index=False)

    final_selected_frames_data = update_coordinates(all_videos_valid_keypoints_path, kmeans_frames_path)

    with open(final_selected_frames_path, 'w', newline='') as f:
        csv.writer(f).writerows(final_selected_frames_data)

    print("Frame selection strategy pipeline completed successfully.")
    return final_selected_frames_path

def process_and_export_frame_selection(
    cfg: dict,
    cfg_lp: dict,
    data_dir: str,
    labeled_data_dir: str,
    final_selected_frames_path: str,
    seed_labels: pd.DataFrame
) -> pd.DataFrame:
    """Process selected frames and export them."""
    selected_frames = pd.read_csv(final_selected_frames_path, header=[0,1,2], index_col=0)
    print(f"Reading data from {final_selected_frames_path}")
    
    standard_scorer_name = 'standard_scorer'
    if not seed_labels.empty:
        seed_labels.columns = pd.MultiIndex.from_arrays([
            [standard_scorer_name] * len(seed_labels.columns),
            seed_labels.columns.get_level_values('bodyparts'),
            seed_labels.columns.get_level_values('coords')
        ], names=['scorer', 'bodyparts', 'coords'])

    for idx, row in selected_frames.iterrows():
        video_name, frame_number = idx.split('/')
        frame_number = int(frame_number)
        new_index = f"labeled-data/{video_name}/img{str(frame_number).zfill(8)}.png"
        new_row = pd.DataFrame(row).T
        new_row.index = [new_index]
        new_row.columns = pd.MultiIndex.from_arrays([
            [standard_scorer_name] * len(new_row.columns),
            new_row.columns.get_level_values('bodyparts'),
            new_row.columns.get_level_values('coords')
        ], names=['scorer', 'bodyparts', 'coords'])
        seed_labels = pd.concat([seed_labels, new_row])

    video_paths = {}
    for video_dir in cfg["video_directories"]:
        video_dir_path = os.path.join(data_dir, video_dir)
        for video_file in os.listdir(video_dir_path):
            if video_file.lower().endswith('.mp4'):
                video_name = os.path.splitext(video_file)[0]
                video_paths[video_name] = os.path.join(video_dir_path, video_file)

    frames_by_video = {}
    for idx in seed_labels.index:
        if idx.startswith("labeled-data/"):
            video_name, frame_name = idx.split('/')[-2:]
            frame_number = int(frame_name.split('.')[0][3:])
            frames_by_video.setdefault(video_name, []).append(frame_number)

    for video_name, frame_numbers in frames_by_video.items():
        if video_name in video_paths:
            export_frames(
                video_file=video_paths[video_name],
                save_dir=os.path.join(labeled_data_dir, video_name),
                frame_idxs=np.array(frame_numbers),
                format="png",
                n_digits=8,
                context_frames=0,
            )
        else:
            print(f"Warning: Video file for {video_name} not found in the specified directories.")

    return seed_labels
