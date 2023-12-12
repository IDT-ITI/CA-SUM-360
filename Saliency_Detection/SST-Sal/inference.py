import os
import numpy as np
import torch

from data_loader import RGB
from models import SST_Sal
from torch.utils.data import DataLoader
import cv2
from metrics import Metrics

from configs import inference_args

current_script_path = os.path.abspath(__file__)

# Navigate to the parent directory (one level up)
parent_directory = os.path.dirname(current_script_path)
grant_parent_directory = os.path.dirname(parent_directory)
grant_parent_directory = os.path.dirname(grant_parent_directory)


def test(test_data, model,load_gt ,device,output_path):


    model.to(device)
    model.eval()
    ccall = []
    simall = []

    with torch.no_grad():
        if load_gt=="True":

            for j,video in enumerate(test_data):

                cc_metric = []
                sim_metric = []

                for k,(x,y) in enumerate(video):


                    pred = model(x.to(device))

                    batch_size, Nframes, _, _ = pred[:, :, 0, :, :].shape

                    if k ==0:

                        for bs in range(0,batch_size):

                            for i in range(0,Nframes):

                                cc,sim= Metrics(pred[bs][i][0].cpu(),y[bs][i][0].cpu())
                                cc_metric.append(cc)
                                sim_metric.append(sim)


                    else:
                        for bs in range(0, batch_size):
                            for i in range(4, Nframes):

                                cc, sim = Metrics(pred[bs][i][0].cpu(), y[bs][i][0].cpu())
                                cc_metric.append(cc)
                                sim_metric.append(sim)

                #print(np.mean(cc_metric))
                #print(np.mean(sim_metric))

                ccall.append(np.mean(cc_metric))
                simall.append(np.mean(sim_metric))

            print(np.mean(ccall))
            print(np.mean(simall))

        else:
            count = 0
            for j, video in enumerate(test_data):
                if os.path.exists(output_path+ "/sst-sal_" + str(j)):
                    print(path_to_save_saliency_maps + " path exists")
                else:
                    os.mkdir(output_path+ "/sst-sal_" + str(j))
                for k, (x) in enumerate(video):

                    pred = model(x.to(device))
                    batch_size, Nframes, _, _ = pred[:, :, 0, :, :].shape

                    if k == 0:

                        for bs in range(0, batch_size):
                            for i in range(0, Nframes):
                                sal = np.array((pred[bs][i][0].cpu() - torch.min(pred[bs][i][0].cpu())) / (
                                        torch.max(pred[bs][i][0].cpu()) - torch.min(pred[bs][i][0].cpu())))
                                cv2.imwrite(os.path.join(output_path+ "/sst-sal_" + str(j), f'{count:04d}.png'), (sal * 255).astype(np.uint8))
                                count += 1


                    else:
                        for bs in range(0, batch_size):
                            for i in range(4, Nframes):
                                sal = np.array((pred[bs][i][0].cpu() - torch.min(pred[bs][i][0].cpu())) / (
                                        torch.max(pred[bs][i][0].cpu()) - torch.min(pred[bs][i][0].cpu())))
                                cv2.imwrite(os.path.join(output_path+ "/sst-sal_" + str(j), f'{count:04d}.png'), (sal * 255).astype(np.uint8))
                                count += 1



if __name__ == "__main__":

    args = inference_args().parse_args()
    gpu = args.gpu
    sst_sal = args.sst_sal
    path_to_frames_folder = args.path_to_ERP_frames
    process = args.process
    resolution = args.resolution
    clip_size = args.clip_size
    batch_size = args.batch_size
    path_to_save_saliency_maps = args.path_to_extracted_saliency_maps
    load_gt = args.load_gt
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    model = torch.load(sst_sal, map_location=device)
    val_path_txt = os.path.join(grant_parent_directory,"data/Static-VR-EyeTracking/val_split.txt")
    with open(val_path_txt, 'r') as file:
        # Read the content of the file and split it by commas
        content = file.read()
        videos = content.split(',')
    static_val_videos = [video.strip() for video in videos]
    path_to_frames_folder = os.path.join(grant_parent_directory,path_to_frames_folder)
    path_to_save_saliency_maps = grant_parent_directory + "/" + path_to_save_saliency_maps
    if os.path.exists(path_to_save_saliency_maps):
        print(path_to_save_saliency_maps + " path exists")
    else:
        path_to_save_saliency_maps = os.mkdir(path_to_save_saliency_maps)
        print("path to save the saliency maps", path_to_save_saliency_maps)

    test_video_dataset = RGB(path_to_frames_folder,static_val_videos,load_gt,process=process,frames_per_data=clip_size,resolution=resolution)
    test_data = DataLoader(test_video_dataset, batch_size=batch_size, drop_last=True)


    test(test_data, model,load_gt, device,path_to_save_saliency_maps)
