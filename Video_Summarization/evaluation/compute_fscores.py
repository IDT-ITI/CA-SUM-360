# -*- coding: utf-8 -*-
from os import listdir
import json
import numpy as np
import h5py
from generate_summary import generate_summary

# arguments to run the script
path = '/home/apostolid/data/Summaries/CA-SUM-TransMIXR/exp4/reg0.8/TransMIXR/results/split0'

results = [f for f in listdir(path) if f.endswith(".json")]
dataset_path = '/home/apostolid/data/summarization_datasets/TransMIXR/transmixr.h5'

best_epoch = '318'

all_scores = []
with open(path + '/TransMIXR_' + best_epoch + '.json') as f:     # read the json file ...
    data = json.loads(f.read())
    keys = list(data.keys())

    for video_name in keys:                    # for each video inside that json file ...
        scores = np.asarray(data[video_name])  # read the importance scores from frames
        all_scores.append(scores)

all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
with h5py.File(dataset_path, 'r') as hdf:
    for video_name in keys:
        sb = np.array(hdf.get(video_name + '/change_points'))
        n_frames = np.array(hdf.get(video_name + '/n_frames'))
        positions = np.array(hdf.get(video_name + '/picks'))

        all_shot_bound.append(sb)
        all_nframes.append(n_frames)
        all_positions.append(positions)

all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

