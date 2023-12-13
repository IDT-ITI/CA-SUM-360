import torch
import os

import numpy as np
import cv2

from torch.utils.data import Dataset


class RGB_salient360_sitzman_loader(Dataset):

    def __init__(self, path_frames, process, frames_per_data):

        self.path = path_frames
        self.process = process
        self.frames_per_data = frames_per_data

        self.video_list = []
        self.dataset = []

        png_files = []


        frames = os.listdir(self.path)
        for j, fram in enumerate(frames):
            if fram.lower().endswith(('.png', '.jpg')):
                png_files.append(self.path + "/" + fram)


        png_files = sorted(png_files)

        for j in range(len(png_files)):
            self.dataset.append(png_files[j])


    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        'Generates one sample of data'

        path_to_frame = self.dataset[idx]


        data = []
        list_of_gt = []
        list_of_fx = []
        path_to_gt = path_to_frame.replace('frames', 'saliency')
        path_to_fx = path_to_frame.replace('frames', 'fixation')

        X = cv2.imread(path_to_frame)
        X = cv2.resize(X, (640, 320))
        X = X.astype(np.float32)
        X -= [103.939, 116.779, 123.68]
        X = torch.FloatTensor(X)
        X = X.permute(2, 0, 1)


        data.append(X)

        img_gt = cv2.imread(path_to_gt, 0)

        img_gt = cv2.resize(img_gt, (640, 320))
        img_gt = img_gt.astype(np.float32)
        img_gt = (img_gt - np.min(img_gt)) / (np.max(img_gt) - np.min(img_gt))
        img_gt = torch.FloatTensor(img_gt)
        img_gt = img_gt.unsqueeze(0)

        list_of_gt.append(img_gt)

        img_fx = cv2.imread(path_to_fx, 0)
        img_fx = cv2.resize(img_fx, (640, 320))
        img_fx = img_fx.astype(np.float32)
        img_fx = (img_fx - np.min(img_fx)) / (np.max(img_fx) - np.min(img_fx))
        img_fx = torch.FloatTensor(img_fx)
        img_fx = img_fx.unsqueeze(0)

        list_of_fx.append(img_fx)

        data_tensor = torch.cat(data, 0)
        gt_tensor = torch.cat(list_of_gt, 0)
        fx_tensor = torch.cat(list_of_fx, 0)

        samples=(data_tensor, gt_tensor, fx_tensor)



        return samples
