import torch
import os
import numpy as np
import cv2

from torch.utils.data import DataLoader, Dataset

class RGB_sports360(Dataset):

    def __init__(self, root_path,video_names,load_gt,frames_per_data):

        self.root_path = root_path
        self.load_gt = load_gt
        self.frames_per_data = frames_per_data
        self.video_list = []
        self.dataset = []

        self.video_names = video_names

        for i, file in enumerate(self.video_names):

            png_files = []

            frame_folder = os.path.join(self.root_path, file)
            frames = os.listdir(frame_folder)
            for j, fram in enumerate(frames):

                # if count % 3 == 0:
                if fram.lower().endswith(('.png', '.jpg')):
                    png_files.append(frame_folder + "/" + fram)
                # count += 1

            png_files = sorted(png_files)

            fpd = self.frames_per_data + 20

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

        frames = self.dataset[idx]


        samples=[]

        for paths_to_frame in frames:
            data = []
            list_of_gt = []
            path_list = []
            for path_to_frame in paths_to_frame:
                directory, filename = os.path.split(path_to_frame)
                file_name = os.path.splitext(filename)[0]
                desired_name = file_name.split('_')[0]
                path_to_gt = os.path.join(directory, f"{desired_name}_gt.npy")
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

                img_gt = np.load(path_to_gt).squeeze()

                img_gt = cv2.resize(img_gt, (640, 320), interpolation=cv2.INTER_AREA)
                img_gt = img_gt.astype(np.float32)
                img_gt = (img_gt - np.min(img_gt)) / (np.max(img_gt) - np.min(img_gt))
                img_gt = torch.FloatTensor(img_gt)

                img_gt = img_gt.unsqueeze(0)
                list_of_gt.append(img_gt.unsqueeze(0))


                data_tensor = torch.cat(data, 0)
                gt_tensor = torch.cat(list_of_gt, 0)


            samples.append((data_tensor, gt_tensor,path_list))

        return samples
