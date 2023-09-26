import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils


# The DataLoader for our specific datataset with extracted frames
class Static_dataset(data.Dataset):

    def __init__(self, split, number_of_frames, root_path, load_gt=True, resolution=None, val_perc=0.2):
        # augmented frames
        self.frames_path = os.path.join(root_path, 'frames')

        self.load_gt = load_gt


            # ground truth
        self.gt_path = os.path.join(root_path, "saliency")
        self.fx_path = os.path.join(root_path, "fixation")

        self.resolution = resolution
        self.frames_list = []
        self.gt_list = []
        self.fx_list = []
        # Gives accurate human readable time, rounded down not to include too many decimals
        print('start load data')
        frame_files = os.listdir(os.path.join(self.frames_path))
        self.frames_list = sorted(frame_files, key=lambda x: int(x.split(".")[0]))
        self.frames_list = self.frames_list[:number_of_frames]
        print(' load images data')

        gt_files = os.listdir(os.path.join(self.gt_path))
        fx_files = os.listdir(os.path.join(self.fx_path ))

        self.gt_list = sorted(gt_files, key=lambda x: int(x.split(".")[0]))
        self.fx_list = sorted(fx_files, key = lambda x: int(x.split(".")[0]) )

        print(' load groundtruth data')
        self.gt_list = self.gt_list[:number_of_frames]
        self.fx_list = self.fx_list[:number_of_frames]

        print('data loaded')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.frames_list)

    def __getitem__(self, frame_index):

        'Generates one sample of data'

        frame = self.frames_list[frame_index]


        gt = self.gt_list[frame_index]
        fx = self.fx_list[frame_index]

        path_to_frame = os.path.join(self.frames_path, frame)

        X = cv2.imread(path_to_frame)
        X = cv2.resize(X,(640,320))

        X = X.astype(np.float32)

        X = X - [103.939, 116.779, 123.68]
        X = torch.cuda.FloatTensor(X)
        X = X.permute(2, 0, 1)
        # add batch dim
        data = X.unsqueeze(0)


        # Load and preprocess ground truth (saliency maps)

        path_to_gt = os.path.join(self.gt_path, gt)

        y = cv2.imread(path_to_gt,0)
        y = cv2.resize(y,(640,320))
        y = y.astype(np.float32)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        y = torch.FloatTensor(y)
        # y = y.permute(2,0,1)
        gt = y.unsqueeze(0)

        path_to_fx = os.path.join(self.fx_path, fx)

        y1 = cv2.imread(path_to_fx, 0)
        y1 = cv2.resize(y1, (640, 320))
        y1 = y1.astype(np.float32)
        y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
        y1 = torch.FloatTensor(y1)

        fx = y1.unsqueeze(0)


        packed = (data, gt,fx)  # pack data with the corresponding  ground truths
        #packed = (data, gt)

        return packed