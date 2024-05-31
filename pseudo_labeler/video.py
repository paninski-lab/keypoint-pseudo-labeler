"""Functions for video handling."""

import cv2
import numpy as np


def get_frames_from_idxs(cap: cv2.VideoCapture, idxs: np.ndarray) -> np.ndarray:
    """Helper function to load video segments.

    Parameters
    ----------
    cap: cv2.VideoCapture object
    idxs: frame indices into video

    Returns
    -------
    np.ndarray
        returned frames of shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype="uint8")
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                "warning! reached end of video; returning blank frames for remainder of "
                + "requested indices"
            )
            break
    return frames
