import torch
import os
import cv2
import math
from torch.utils.data import Dataset
import numpy as np


class RGB(Dataset):
    def __init__(self, path_to_frames,static_videos, process,frames_per_data,resolution=[240, 320]):
        self.path_to_frames = path_to_frames
        self.video_names = static_videos
        self.resolution = resolution
        self.dataset = []
        self.frames_per_data = frames_per_data
        self.process = process
        #video_names = os.listdir(self.path_to_frames)
        if self.process=="train":
            for i, file in enumerate(self.video_names):
                frame_folder = os.path.join(self.path_to_frames, file)

                if os.path.exists(frame_folder):

                    img_files = []


                    frames = os.listdir(frame_folder)

                    for j, fram in enumerate(frames):
                        if fram.lower().endswith(('.png','.jpg')):
                            img_files.append(frame_folder + "/" + fram)


                    img_files = sorted(img_files)
                    fpd = 20 + self.frames_per_data

                    for j in range(20, len(img_files),self.frames_per_data):

                        self.dataset.append(img_files[j:fpd])
                        fpd = fpd+ self.frames_per_data

        else:
            for i, file in enumerate(self.video_names):

                frame_folder = os.path.join(self.path_to_frames, file)
                print(frame_folder)
                if os.path.exists(frame_folder):

                    img_files = []


                    frames = os.listdir(frame_folder)

                    for j, fram in enumerate(frames):
                        if fram.lower().endswith(('.png', '.jpg')):
                            img_files.append(frame_folder + "/" + fram)


                    img_files = sorted(img_files)
                    fpd = self.frames_per_data+20
                    data=[]
                    for j in range(20, len(img_files), self.frames_per_data-4): #apply this -4 factor to have smoother results in the final saliency maps, as SST-Sal predicts in sequences of 20 frames
                        data.append(img_files[j:fpd])

                        fpd = fpd + self.frames_per_data-4
                    self.dataset.append(data)



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frames = self.dataset[idx]

        frame_img = []
        label = []
        final = []
        if self.process =="train":
            for i, path_to_frame in enumerate(frames):



                path_to_gt = path_to_frame.replace('frames', 'saliency')


                img_frame = cv2.imread(path_to_frame)

                if img_frame.shape[1] != self.resolution[1] or img_frame.shape[0] != self.resolution[0]:
                    img_frame = cv2.resize(img_frame, (self.resolution[1], self.resolution[0]),
                                           interpolation=cv2.INTER_AREA)
                img_frame = img_frame.astype(np.float32)

                img_frame = img_frame / 255

                img_frame = torch.FloatTensor(img_frame)
                img_frame = img_frame.permute(2, 0, 1)
                img_frame = img_frame.unsqueeze(0)
                frame_img.append(img_frame)

                img_gt = cv2.imread(path_to_gt, cv2.IMREAD_GRAYSCALE)
                if img_gt.shape[1] != self.resolution[1] or img_gt.shape[0] != self.resolution[0]:
                    img_gt = cv2.resize(img_gt, (self.resolution[1], self.resolution[0]),
                                        interpolation=cv2.INTER_AREA)
                img_gt = img_gt.astype(np.float32)
                img_gt = (img_gt - np.min(img_gt)) / (np.max(img_gt) - np.min(img_gt))
                img_gt = torch.FloatTensor(img_gt)
                img_gt = img_gt.unsqueeze(0)
                label.append(img_gt.unsqueeze(0))

            final.append((torch.cat(frame_img, 0), torch.cat(label, 0)))
        else:
            for i, path_to_frames in enumerate(frames):
                frame_img = []
                label = []

                for path_to_frame in path_to_frames:

                    path_to_gt = path_to_frame.replace('frames', 'saliency')

                    img_frame = cv2.imread(path_to_frame)

                    if img_frame.shape[1] != self.resolution[1] or img_frame.shape[0] != self.resolution[0]:
                        img_frame = cv2.resize(img_frame, (self.resolution[1], self.resolution[0]),
                                               interpolation=cv2.INTER_AREA)
                    img_frame = img_frame.astype(np.float32)

                    img_frame = img_frame / 255

                    img_frame = torch.FloatTensor(img_frame)
                    img_frame = img_frame.permute(2, 0, 1)
                    img_frame = img_frame.unsqueeze(0)
                    frame_img.append(img_frame)

                    img_gt = cv2.imread(path_to_gt, cv2.IMREAD_GRAYSCALE)
                    if img_gt.shape[1] != self.resolution[1] or img_gt.shape[0] != self.resolution[0]:
                        img_gt = cv2.resize(img_gt, (self.resolution[1], self.resolution[0]),
                                            interpolation=cv2.INTER_AREA)
                    img_gt = img_gt.astype(np.float32)
                    img_gt = (img_gt - np.min(img_gt)) / (np.max(img_gt) - np.min(img_gt))
                    img_gt = torch.FloatTensor(img_gt)
                    img_gt = img_gt.unsqueeze(0)
                    label.append(img_gt.unsqueeze(0))

                final.append((torch.cat(frame_img, 0), torch.cat(label, 0)))

        return final
