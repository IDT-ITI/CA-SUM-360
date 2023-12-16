# -*- coding: utf-8 -*-
import numpy as np
import csv
import json
import sys

# example usage: python evaluation/choose_best_model.py 1 "360VideoSumm"
exp_num = sys.argv[1]
dataset = sys.argv[2]

base_path = ".../CA-SUM-360/Summaries"


def get_improvement_score(epoch, reg_factor):
    """ Using the estimated frame-level importance scores from an untrained model, calculate the improvement (eq. 2-3)
    of a  trained model for the chosen epoch, on a given split and regularization factor.

    :param int epoch: The chosen training epoch for the given split and regularization factor.
    :param float reg_factor: The value of the current evaluated length regularization factor
    :return: The relative improvement of a trained model over an untrained (random) one.
    """
    untr_path = f"{base_path}/exp{exp_num}/reg{reg_factor}/{dataset}/results/split0/{dataset}_-1.json"
    curr_path = f"{base_path}/exp{exp_num}/reg{reg_factor}/{dataset}/results/split0/{dataset}_{epoch}.json"
    with open(curr_path) as curr_file, open(untr_path) as untr_file:
        untr_data = json.loads(untr_file.read())
        curr_data = json.loads(curr_file.read())

        keys = list(curr_data.keys())
        mean_untr_scores, mean_curr_scores = [], []
        for video_name in keys:                              # For a video inside that split get the ...
            untr_scores = np.asarray(untr_data[video_name])  # Untrained model computed importance scores
            curr_scores = np.asarray(curr_data[video_name])  # trained model computed importance scores

            mean_untr_scores.append(np.mean(untr_scores))
            mean_curr_scores.append(np.mean(curr_scores))

    mean_untr_scores = np.array(mean_untr_scores)
    mean_curr_scores = np.array(mean_curr_scores)

    # Measure how much did we improve a random model, relatively to moving towards sigma (minimum loss)
    improvement = abs(mean_curr_scores.mean() - mean_untr_scores.mean())
    result = (improvement / abs(reg_factor - mean_untr_scores.mean()))

    return result


def train_logs(log_file, method="argmin"):
    """ Choose and return the epoch based only on the training loss. Through the `method` argument you can get the epoch
    associated with the minimum training loss (argmin) or the last epoch of the training process (last).

    :param str log_file: Path to the saved csv file containing the loss information.
    :param str method: The chosen criterion for the epoch (model) picking process.
    :return: The epoch of the best model, according to the chosen criterion.
    """
    losses = {}
    losses_names = []

    # Read the csv file with the training losses
    with open(log_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                for col in range(len(row)):
                    losses[row[col]] = []
                    losses_names.append(row[col])
            else:
                for col in range(len(row)):
                    losses[losses_names[col]].append(float(row[col]))

    # criterion: The length regularization of the generated summary (400 epochs, after which overfitting problems occur)
    loss = losses["loss_epoch"]
    loss = loss[:400]
    START_EPOCH = 20                      # If unstable training is observed at the start

    if method == "last":
        epoch = len(loss) - 1
    elif method == "argmin":
        epoch = np.array(loss[START_EPOCH:]).argmin() + START_EPOCH
    else:
        raise ValueError(f"Method {method} is not currently supported. Only `last` and `argmin` are available.")

    return epoch


# Choose the model associated with the min training loss for each regularization factor and get its improvement score
all_improvements, all_epochs = [], []
sigmas = [i/10 for i in range(5, 10)]  # The valid values for the length regularization factor
for sigma in sigmas:
    split_improvements, split_epochs = np.zeros(1, dtype=float), np.zeros(1, dtype=int)
    for split in range(0, 1):
        log = f"{base_path}/exp{exp_num}/reg{sigma}/{dataset}/logs/split{split}/scalars.csv"
        selected_epoch = train_logs(log, method="argmin")  # w/o +1. (only needed to pick the f-score value)

        split_improvements[split] = get_improvement_score(epoch=selected_epoch, reg_factor=sigma)
        split_epochs[split] = selected_epoch
    all_improvements.append(split_improvements)
    all_epochs.append(split_epochs)

# From list to nd array for easier computations
all_improvements = np.stack(all_improvements)
all_epochs = np.stack(all_epochs)

# Choose the highest improvement sigma's per split
all_improvements = np.where(all_improvements > 1.5, 0, all_improvements)
# print(all_improvements)
improvement_per_spit = all_improvements.max(axis=0, initial=-1)
chosen_indices = all_improvements.argmax(axis=0)
sigma_per_split = np.array(sigmas)[chosen_indices]

# For the chosen epochs and length regularization factors, calculate the metrics for our assessments
curr_sigma = sigma_per_split[0]
curr_epoch = all_epochs[chosen_indices[0], 0] + 1  # because of the evaluation on the untrained model

print(f" [\u03C3={curr_sigma}, epoch: {curr_epoch}]")
