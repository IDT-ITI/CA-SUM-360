# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.filename = '/home/apostolid/data/summarization_datasets/TransMIXR/transmixr.h5'
        self.splits_filename = ['/home/apostolid/data/summarization_datasets/TransMIXR/' + self.name + '_split.json']
        self.split_index = split_index

        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features = []
        self.saliency_scores = []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
            sal_scores = torch.Tensor(np.array(hdf[video_name + '/saliency_scores']))

            self.list_frame_features.append(frame_features)
            self.saliency_scores.append(sal_scores)

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len
    
    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features
        test  mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        saliency_scores = self.saliency_scores[index]
        return frame_features, video_name, saliency_scores


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass
