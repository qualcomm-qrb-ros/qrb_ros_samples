# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Pre-/post-processing for the stage-3 gesture classifier.
# The logic follows Qualcomm's official quic/ai-hub-models:
#   src/qai_hub_models/models/_shared/mediapipe/utils.py :: preprocess_hand_x64
#   src/qai_hub_models/models/mediapipe_hand_gesture/model.py :: GESTURE_LABELS
# It is reimplemented here in numpy (the board has no torch).

import numpy as np

# The 8 gesture labels; the order matches the model output Identity[1,8] index
# (the official GESTURE_LABELS).
GESTURE_LABELS = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]

# Anatomical anchors used for normalization: the wrist plus the finger bases.
CENTER_IDX = np.array([0, 1, 5, 9, 13, 17], dtype=np.int64)
_EPS = 1e-5


def preprocess_hand_x64(pts, handedness, mirror=False):
    """Assemble the 21 keypoints + handedness into the classifier input [1,64].

    This mirrors the official preprocess_hand_x64 (numpy version).

    Parameters
    ----------
    pts : np.ndarray, shape (21, 3) or (1, 21, 3)
        The local landmarks (x, y, z) from the landmark detector.
    handedness : float or np.ndarray
        The stage-2 lr output (left/right-hand score in [0, 1]).
    mirror : bool
        When True, build the mirrored hand (negate x, handedness = 1 - handedness).

    Returns
    -------
    np.ndarray, shape (1, 64), dtype float32
        First 63 = normalized 21x3 landmarks, element 64 = handedness scalar.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(21, 3).copy()
    h = float(np.asarray(handedness).reshape(-1)[0])

    if mirror:
        pts[:, 0] *= -1.0        # negate x only
        h = 1.0 - h              # flip handedness

    # Translate so the mean of the 6 anatomical anchors sits at the origin.
    center = pts[CENTER_IDX, :].mean(axis=0, keepdims=True)  # (1,3)
    normed = pts - center

    # Scale by the larger of the x / y spans.
    range_x = normed[:, 0].max() - normed[:, 0].min()
    range_y = normed[:, 1].max() - normed[:, 1].min()
    scale = max(range_x, range_y) + _EPS

    pts_n = normed / scale
    flat = pts_n.reshape(63)                                 # flatten 21x3

    x64 = np.concatenate([flat, np.array([h], dtype=np.float32)])  # 63 + 1
    return x64.reshape(1, 64).astype(np.float32)
