# -*- coding: utf-8 -*-
import numpy as np
from knapsack_implementation import knapSack


def generate_summary(shot_bound, scores, nframes, positions):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] shot_bound: The video shots for the -original- testing video.
    :param list[np.ndarray] scores: The calculated frame importance scores for the sub-sampled testing video.
    :param list[np.ndarray] nframes: The number of frames for the -original- testing video.
    :param list[np.ndarray] positions: The position of the sub-sampled frames for the -original- testing video.
    """

    # Get shots' boundaries
    shot_bound = shot_bound[0]    # [number_of_shots, 2]
    frame_init_scores = scores[0]
    n_frames = nframes[0]
    positions = positions[0]

    # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(frame_init_scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = frame_init_scores[i]

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_shot = shot_bound[-1]
    final_max_length = int((final_shot[1] + 1) * 0.15)

    selected_fragments = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    selected_frames = np.zeros((selected_fragments.__len__(), 2), dtype=np.int64)

    for count, shot in enumerate(selected_fragments):
        selected_frames[count][0] = shot_bound[shot][0] + 1
        selected_frames[count][1] = shot_bound[shot][1] + 1

    return selected_fragments, selected_frames.tolist()
