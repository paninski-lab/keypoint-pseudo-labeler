"""Function to select pseudo-labeled frames."""

import os

import cv2
import numpy as np
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


def select_frame_idxs_eks(
    video_file: str,
    n_frames_to_select: int,
    seed: int
) -> np.ndarray:
    total_frames = get_total_frames(video_file)
    np.random.seed(seed)
    return np.random.choice(total_frames, n_frames_to_select, replace=False)



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
    
