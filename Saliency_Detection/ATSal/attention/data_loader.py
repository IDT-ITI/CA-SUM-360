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


class RGB_dataset(Dataset):

    def __init__(self, root_path,process,frames_per_data):

        self.root_path = root_path
        self.process = process
        self.frames_per_data = frames_per_data

        self.video_list = []
        self.dataset = []
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

                for j in range(20, len(png_files)):

                    #datast.append(png_files[j:fpd])
                    self.dataset.append(png_files[j])
                    #fpd = fpd + frames_per_data

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

                fpd = self.frames_per_data+20

                dat = []

                for j in range(20, len(png_files), frames_per_data):

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
        path_to_frame = self.dataset[idx]
        frames = self.dataset[idx]
        #print(frames)

        samples=[]
        if self.process =="train":

            data = []
            list_of_gt = []
            list_of_fx = []
            #print(frames)
            #for path_to_frame in frames:

            #for path_to_frame in paths_to_frame:
            #print(path_to_frame)

            path_to_gt = path_to_frame.replace('frames', 'saliency')
            path_to_fx = path_to_frame.replace('frames', 'fixation')

            #path_list.append(path_to_frame)

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

            img_fx = cv2.imread(path_to_fx, 0)
            # img_gt = cv2.resize(img_gt,(640,320))

            img_fx = cv2.resize(img_fx, (640, 320))
            img_fx = img_fx.astype(np.float32)
            img_fx = (img_fx - np.min(img_fx)) / (np.max(img_fx) - np.min(img_fx))
            img_fx = torch.FloatTensor(img_fx)

            img_fx = img_fx.unsqueeze(0)
            list_of_fx.append(img_fx.unsqueeze(0))

            data_tensor = torch.cat(data, 0)
            gt_tensor = torch.cat(list_of_gt, 0)
            fx_tensor = torch.cat(list_of_fx,0)

            samples.append((data_tensor, gt_tensor,fx_tensor))


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