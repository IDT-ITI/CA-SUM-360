import torch
import os
import datetime
import numpy as np
import cv2

from torch.utils.data import DataLoader, Dataset

from torchvision import utils
from tqdm import tqdm
import torch as th
from torch import nn


class Multiexpert_dataset(Dataset):

    def __init__(self, root_path,process,frames_per_data,resolution):

        self.root_path = root_path
        self.process = process
        self.frames_per_data = frames_per_data
        self.resolution = resolution
        self.video_list = []
        self.dataset = []
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        video_names = os.listdir(root_path)
        if self.process=="train":

            for i, file in enumerate(video_names):
            
                png_files = []

                frame_folder = os.path.join(self.root_path, file)
                frames = os.listdir(frame_folder)
                for j, fram in enumerate(frames):

                    # if count % 3 == 0:
                    if fram.lower().endswith(('.png', '.jpg')):
                        png_files.append(frame_folder + "/" + fram)
                    # count += 1

                png_files = sorted(png_files)

                fpd = self.frames_per_data+20  #skip first 20 frames for train process because observers were exploring the 360 video from a fixed starting point

                datast = []

                for j in range(20, len(png_files),frames_per_data):

                    #datast.append(png_files[j:fpd])
                    self.dataset.append(png_files[j:fpd])
                    fpd = fpd + frames_per_data

                    #self.dataset.append(datast)
        else:
            for i, file in enumerate(video_names):

                png_files = []

                frame_folder = os.path.join(self.root_path, file)
                frames = os.listdir(frame_folder)
                for j, fram in enumerate(frames):

                    # if count % 3 == 0:
                    if fram.lower().endswith(('.png', '.jpg')):
                        png_files.append(frame_folder + "/" + fram)
                    # count += 1

                png_files = sorted(png_files)

                fpd = self.frames_per_data

                dat = []

                for j in range(0, len(png_files), frames_per_data):

                    dat.append(png_files[j:fpd])
                    fpd = fpd + frames_per_data

                self.dataset.append(dat)




    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        'Generates one sample of data'
        #print(video_index)
        frames = self.dataset[idx]
        #print(frames)

        samples=[]
        if self.process =="train":

            data = []
            list_of_gt = []

            #print(frames)
            for path_to_frame in frames:
                #print(path_to_frame)

                #for path_to_frame in paths_to_frame:
                #print(path_to_frame)

                path_to_gt = path_to_frame.replace('frames', 'saliency')

                X = cv2.imread(path_to_frame)
                # if self.resolution!=None:
                X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
                X = X.astype(np.float32)
                X -= self.ImageNet_mean
                # X = (X-np.min(X))/(np.max(X)-np.min(X))
                X = torch.FloatTensor(X)
                X = X.permute(2, 0, 1)  # swap channel dimensions

                data.append(X.unsqueeze(0))
                # Load and preprocess ground truth (saliency maps)




                y = cv2.imread(path_to_gt, 0)  # Load as grayscale
                # if self.resolution!=None:
                y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
                if y.max() != 0.0:
                    y = (y - np.min(y)) / (np.max(y) - np.min(y))
                y = torch.FloatTensor(y)

                list_of_gt.append(y.unsqueeze(0))
            data_tensor = torch.cat(data, 0)
            gt_tensor = torch.cat(list_of_gt, 0)

            samples.append((data_tensor, gt_tensor))


        else:

            for paths_to_frame in frames:
                data = []
                list_of_gt = []

                path_list = []
                for path_to_frame in paths_to_frame:


                    path_to_gt = path_to_frame.replace('frames', 'saliency')


                    path_list.append(path_to_frame)

                    X = cv2.imread(path_to_frame)

                    X = cv2.resize(X, (640, 320))
                    X = X.astype(np.float32)

                    X -= [103.939, 116.779, 123.68]

                    X = torch.FloatTensor(X)
                    X = X.permute(2, 0, 1)
                    data1 = X.unsqueeze(0)
                    #
                    data.append(data1)

                    img_gt = cv2.imread(path_to_gt, 0)
                    # img_gt = cv2.resize(img_gt,(640,320))

                    img_gt = cv2.resize(img_gt, (640, 320))
                    img_gt = img_gt.astype(np.float32)
                    img_gt = (img_gt - np.min(img_gt)) / (np.max(img_gt) - np.min(img_gt))
                    img_gt = torch.FloatTensor(img_gt)

                    img_gt = img_gt.unsqueeze(0)
                    list_of_gt.append(img_gt.unsqueeze(0))



                data_tensor = torch.cat(data, 0)
                gt_tensor = torch.cat(list_of_gt, 0)


                samples.append((data_tensor, gt_tensor,path_list))

        return samples

# video_name_list = dataset.video_names()


# self.video_list.append(frame_files_sorted)
# self.video_name_list.append(i)
