# -*- coding: utf-8 -*-
import torch
import numpy as np
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary, get_corr_coeff
from layers.summarizer import CA_SUM
from os import listdir
from os.path import isfile, join
import h5py
import json
import argparse


def str2bool(v):
    """ Transcode string to boolean.

    :param str v: String to be transcoded.
    :return: The boolean transcoding of the string.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def inference(model, data_path, keys):
    """ Used to inference a pretrained `model` on the `keys` test videos, based on the `eval_method` criterion; using
        the dataset located in `data_path'.

        :param nn.Module model: Pretrained model to be inferenced.
        :param str data_path: File path for the dataset in use.
        :param list keys: Containing the test video keys of the used data split.
    """
    model.eval()

    for video in keys:
        with h5py.File(data_path, "r") as hdf:
            # Input features for inference
            frame_features = torch.Tensor(np.array(hdf[f"{video}/features"])).view(-1, 1024)
            frame_features = frame_features.to(model.linear_1.weight.device)

            # Input need for evaluation

            sb = np.array(hdf[f"{video}/change_points"])
            n_frames = np.array(hdf[f"{video}/n_frames"])
            positions = np.array(hdf[f"{video}/picks"])

        with torch.no_grad():
            scores, _ = model(frame_features)  # [1, seq_len]
            scores = scores.squeeze(0).cpu().numpy().tolist()
            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]

    return summary


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # arguments to run the script

    # Model data
    model_path = f"/home/apostolid/data/pretrained_models/CA-SUM/TVSum/split4"
    model_file = [f for f in listdir(model_path) if isfile(join(model_path, f))]

    # Read current split
    split_file = f"/home/apostolid/data/summarization_datasets/TransMIXR/transmixr_split.json"
    with open(split_file) as f:
        data = json.loads(f.read())
        test_keys = data[0]["test_keys"]

    # Dataset path
    dataset_path = f"/home/apostolid/data/summarization_datasets/TransMIXR/transmixr.h5"

    # Create model with paper reported configuration
    trained_model = CA_SUM(input_size=1024, output_size=1024, block_size=60).to(device)
    trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))
    inference(trained_model, dataset_path, test_keys)
