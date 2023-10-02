import os
import numpy as np

import cv2
from models import Sal_based_Attention_module, SalEMA

import projection_methods

from PIL import Image
import torch
from metrics import Metrics

from data_loader import RGB_dataset
from configs import inference_args

from torch.utils import data
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def load_expert(pt_model, new_model,device):
    temp = torch.load(pt_model,map_location=device)['state_dict']

    new_model.load_state_dict(temp)
    return new_model
def load_attention(pt_model, new_model,device):
    temp = torch.load(pt_model,map_location=device)

    new_model.load_state_dict(temp)
    return new_model

counter_val =0
cc_metric = []
sim_metric = []

def test(loader,model,output_path,load_gt):


    model['attention'].to(device)
    model['attention'].eval()


    model['poles'].to(device)
    model['poles'].eval()

    model['equator'].to(device)
    model['equator'].eval()




    counter = 0

    for i, video in enumerate(loader):

        count_cc = []
        count_sim = []

        out_state = {'F': None, 'R': None, 'B': None, 'L': None, 'U': None, 'D': None}
        if load_gt=="True":
            for j, (clip,sal,frames) in enumerate(video):


                clip = clip.type(torch.cuda.FloatTensor).transpose(0,1)

                sal = sal.type(torch.FloatTensor).transpose(0,1)
                out_state = {'F': None, 'R': None, 'B': None, 'L': None, 'U': None, 'D': None}

                for idx in range(clip.size()[0]):



                    first_stream,r = model['attention'](clip[idx])


                    img = np.array(Image.open(frames[idx][0]).resize((640, 320)))

                    if len(img.shape) == 2:
                        img = img[..., None]

                    out = projection_methods.e2c(img, face_w=160, mode='bilinear', cube_format='dict')
                    out_predict = {}
                    for face_key in out:
                        cmp_face = out[face_key].astype(np.float32)
                        state = out_state[face_key]
                        if len(cmp_face.shape) == 2:
                            cmp_face = cmp_face[..., None]
                        cmp_face -= [103.939, 116.779, 123.68]
                        cmp_face = torch.cuda.FloatTensor(cmp_face)
                        cmp_face = cmp_face.permute(2, 0, 1)
                        cmp_face = cmp_face.unsqueeze(0)
                        if face_key == 'U' or face_key == 'D':
                            state, out_face = model['poles'].forward(input_=cmp_face, prev_state=state)
                        else:
                            state, out_face = model['equator'].forward(input_=cmp_face, prev_state=state)

                        state = repackage_hidden(state)
                        out_face = out_face.squeeze()
                        out_face = out_face.cpu().numpy()
                        out_face = out_face * 255 / out_face.max()
                        out_face = cv2.resize(out_face, (160, 160))

                        if len(out_face.shape) == 2:
                            out_face = out_face[..., None]
                        out_predict[face_key] = np.array(out_face.astype(np.uint8))
                        out_state[face_key] = state

                    second_stream = (attention.projection_methods.c2e(out_predict, h=320, w=640, mode='bilinear', cube_format='dict')).reshape(320, 640)

                    second_stream = torch.from_numpy(second_stream/255)

                    first_stream = first_stream[0][0].cpu()
                    saliency_map = first_stream*second_stream
                    cc, sim = Metrics(saliency_map, sal[idx][0][0].cpu())

                    count_cc.append(cc)
                    count_sim.append(sim)

                    x = saliency_map.numpy()
                    x = (x- np.min(x)) / (np.max(x) - np.min(x))
                    x= (x * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(output_path, f"{counter:04d}.png"),x)

                    counter+=1

            print("Video",i)
            print("Expert CC", np.mean(count_cc))
            print("Expert SIM", np.mean(count_sim))


            cc_metric.append(np.mean(count_cc))
            sim_metric.append(np.mean(count_sim))
        else:
            for j, (clip, frames) in enumerate(video):

                clip = clip.type(torch.cuda.FloatTensor).transpose(0, 1)


                out_state = {'F': None, 'R': None, 'B': None, 'L': None, 'U': None, 'D': None}

                for idx in range(clip.size()[0]):

                    first_stream, r = model['attention'](clip[idx])

                    img = np.array(Image.open(frames[idx][0]).resize((640, 320)))

                    if len(img.shape) == 2:
                        img = img[..., None]

                    out = projection_methods.e2c(img, face_w=160, mode='bilinear', cube_format='dict')
                    out_predict = {}
                    for face_key in out:
                        cmp_face = out[face_key].astype(np.float32)
                        state = out_state[face_key]
                        if len(cmp_face.shape) == 2:
                            cmp_face = cmp_face[..., None]
                        cmp_face -= [103.939, 116.779, 123.68]
                        cmp_face = torch.cuda.FloatTensor(cmp_face)
                        cmp_face = cmp_face.permute(2, 0, 1)
                        cmp_face = cmp_face.unsqueeze(0)
                        if face_key == 'U' or face_key == 'D':
                            state, out_face = model['poles'].forward(input_=cmp_face, prev_state=state)
                        else:
                            state, out_face = model['equator'].forward(input_=cmp_face, prev_state=state)

                        state = repackage_hidden(state)
                        out_face = out_face.squeeze()
                        out_face = out_face.cpu().numpy()
                        out_face = out_face * 255 / out_face.max()
                        out_face = cv2.resize(out_face, (160, 160))

                        if len(out_face.shape) == 2:
                            out_face = out_face[..., None]
                        out_predict[face_key] = np.array(out_face.astype(np.uint8))
                        out_state[face_key] = state

                    second_stream = (attention.projection_methods.c2e(out_predict, h=320, w=640, mode='bilinear',
                                                                      cube_format='dict')).reshape(320, 640)

                    second_stream = torch.from_numpy(second_stream / 255)

                    first_stream = first_stream[0][0].cpu()
                    saliency_map = first_stream * second_stream


                    x = saliency_map.numpy()
                    x = (x - np.min(x)) / (np.max(x) - np.min(x))
                    x = (x * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(output_path, f"{counter:04d}.png"), x)

                    counter += 1




if __name__ =="__main__":
    torch.cuda.empty_cache()
    parser = inference_args()
    args = parser.parse_args()

    #Models args
    gpu = args.gpu
    attention_model = args.attention_model
    expertPoles_model = args.expertPoles_model
    expertEquator_model = args.expertEquator_model

    #Dataloader args
    path_to_frames_folder = args.path_to_frames_folder
    load_gt = args.load_gt
    resolution = args.resolution
    clip_size = args.clip_size
    batch_size = args.batch_size

    #path to save saliency maps
    path_to_save_saliency_maps = args.path_to_save_saliency_maps

    att_model = Sal_based_Attention_module()
    salema_copie = SalEMA()
    salema_copie1 = SalEMA()
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    # load weight
    att_model = load_attention(attention_model, att_model,device).cuda()

    Poles = load_expert(expertPoles_model, salema_copie1,device).cuda()
    Equator = load_expert(expertEquator_model, salema_copie,device).cuda()

    model = {"attention": att_model, "poles": Poles, "equator": Equator}

    #path_to_frames_folder = r"D:\Program Files\IoanProjects\StaticVRval\frames"
    #path_to_save_saliency_maps= "attention/outputs" # outputpath for saliency maps

    train_set = RGB_dataset(path_to_frames_folder,load_gt=load_gt,frames_per_data=clip_size)
    loader = data.DataLoader(train_set, batch_size=batch_size)
    with torch.no_grad():

        test(loader, model,output_path=path_to_save_saliency_maps,load_gt=load_gt)





