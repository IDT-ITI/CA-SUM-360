import os
import numpy as np
from model import Sal_based_Attention_module

import torch

from data_loader import RGB_dataset


from LossFunction import LOSS
from torch.utils import data
from configs import training_args
current_script_path = os.path.abspath(__file__)

# Navigate to the parent directory (one level up)
parent_directory = os.path.dirname(current_script_path)
parent_directory = os.path.dirname(parent_directory)
grant_parent_directory = os.path.dirname(parent_directory)
grant_parent_directory = os.path.dirname(grant_parent_directory)
def load_attention(pt_model, new_model, device):
    temp = torch.load(pt_model, map_location=device)

    new_model.load_state_dict(temp)
    return new_model



def train(train_loader,validation_loader,optimizer,criterion, model,device,epochs,saved_model_path):
    model.train()
    model.to(device=device)

    criterion.cuda()
    print("Training process starts")
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        losses = []
        nss_batch = []
        avg_loss = 0
        avg_nss = 0
        counter = 0
        for i, video in enumerate(train_loader):
            for j,(frames,gtruth,fixation) in enumerate(video):

                #print(frames.shape)
                #frames, gtruth, fixation = batch
                frames = frames.type(torch.cuda.FloatTensor)
                gtruth = gtruth
                fixation = fixation

                loss = torch.tensor(0)
                _nss = 0
                optimizer.zero_grad()
                counter += 1
                for r in range(frames.size()[0]):
                    saliency_map, fixation_module = model(frames[r])
                    # saliency_map, fixation_module = model(frame[r])
                    saliency_map = saliency_map.squeeze(0)
                    fixation_module = fixation_module.squeeze(0)
                    total_loss = criterion(saliency_map, gtruth[r][0].to(device), fixation[r][0].to(device), fixation_module)

                    loss = loss.item() + total_loss


                loss.backward()
                optimizer.step()

                avg_loss += loss.data / frames.size()[0]

            losses.append(avg_loss.cpu() / counter)

        print(f"train loss for epoch {epoch}  ",np.mean(losses))
        model.eval()
        val_losses = []
        nss_batch = []
        val_avg_loss = 0
        val_avg_nss = 0
        val_counter = 0
 
        for i, video in enumerate(validation_loader):


            for j, (frames, gtruth, fixation) in enumerate(video):


                frames = frames.type(torch.cuda.FloatTensor)
                gtruth = gtruth
                fixation = fixation

                loss = torch.tensor(0)


                val_counter += 1
                with torch.no_grad():
                    for r in range(frames.size()[0]):
                        saliency_map, fixation_module = model(frames[r])
                        # saliency_map, fixation_module = model(frame[r])
                        saliency_map = saliency_map.squeeze(0)
                        fixation_module = fixation_module.squeeze(0)
                        total_loss = criterion(saliency_map, gtruth[r][0].to(device), fixation[r][0].to(device), fixation_module)

                        loss = loss.item() + total_loss




                    val_avg_loss += loss.data / frames.size()[0]

                val_losses.append(val_avg_loss.cpu() / val_counter)

        print(f'Epoch {epoch} finished with')
        print("Train loss", np.mean(losses))
        print("Val loss", np.mean(val_losses))
        torch.save(model.state_dict(), saved_model_path+"/"+f'attention_{epoch}.pt')






if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    args = training_args().parse_args()
    gpu = args.gpu
    attention_model_path = args.attention_model
    path_to_frames = args.path_to_frames_folder

    process = args.process
    resolution = args.resolution
    clip_size= args.clip_size
    batch_size = args.batch_size
    path_to_save_weights = args.path_to_save_weights
    epochs = args.epochs
    lr =args.lr
    weight_decay = args.weight_decay


    att_model = Sal_based_Attention_module()

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    print("The model will be running on", device, "device")

    attention_model_path =os.path.join(parent_directory, attention_model_path)

    model = load_attention(attention_model_path, att_model,device)

    val_path_txt = os.path.join(grant_parent_directory, "data/VR-EyeTracking/validation_data_split.txt")
    train_path_txt = os.path.join(grant_parent_directory, "data/VR-EyeTracking/training_data_split.txt")
    with open(train_path_txt, 'r') as file:
        content = file.read()
        values = content.split(',')
    train_videos = [value.strip() for value in values]
    with open(val_path_txt, 'r') as file:
        content = file.read()
        values = content.split(',')
    val_videos = [value.strip() for value in values]

    path_to_train_frames = os.path.join(grant_parent_directory,path_to_frames)
    train_set = RGB_dataset(path_to_train_frames,train_videos, process=process, frames_per_data=clip_size)
    train_loader = data.DataLoader(train_set, batch_size=batch_size,shuffle=True,drop_last=True)

    validation_set = RGB_dataset(path_to_frames,val_videos, process=process, frames_per_data=clip_size)
    validation_loader = data.DataLoader(validation_set, batch_size=batch_size,drop_last=True)

    optimizer = torch.optim.Adam(att_model.parameters(),lr=lr,weight_decay=weight_decay)
    criterion = LOSS()

    path_to_save_weights = os.path.join(grant_parent_directory,path_to_save_weights)

    train(train_loader,validation_loader,optimizer,criterion,model,device,epochs=epochs,saved_model_path=path_to_save_weights)
