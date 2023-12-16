# -*- coding: utf-8 -*-
# The script is taking as granted that the experiments are saved under "base_path" as "base_path/exp$EXP_NUM".
# Run the evaluation script of the `exp_num` experiment for a specific `dataset` and a `regularization factor`.
# First, get the training loss from tensorboard as a csv file, for each data-split for the given regularization factor.
# Then, compute the fscore (txt file) associated with the above mentioned data-splits and regularization factor.

base_path=".../CA-SUM-360/Summaries/"
exp_num=$1
dataset=$2
factor=$3

exp_path="$base_path/exp$exp_num/reg$factor"; echo "$exp_path"  # add factor to the path of the experiment

path="$exp_path/$dataset/logs/split0"
python evaluation/exportTensorFlowLog.py "$path" "$path"
