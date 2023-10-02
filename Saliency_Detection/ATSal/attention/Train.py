import os
import numpy as np
from model import Sal_based_Attention_module

import torch

from data_loader import RGB_dataset


from LossFunction import LOSS
from torch.utils import data




def load_attention(pt_model, new_model, device):
    temp = torch.load("weights/" + pt_model + '.pt', map_location=device)

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

        print(f'Epoch {epoch+1} finished with')
        print("Train loss", np.mean(losses))
        print("Val loss", np.mean(val_losses))
        torch.save(model.state_dict(), saved_model_path+"/"+f'attention_{epoch}.pt')






if __name__ == "__main__":
    torch.cuda.empty_cache()

    att_model = Sal_based_Attention_module()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load weight


    print("The model will be running on", device, "device")


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # load weight
    model = load_attention("train_attention_scratch_37", att_model,device)
    train_folder = r"D:\Program Files\IoanProjects\Vrdata3\frames"
    val_folder = r"D:\Program Files\IoanProjects\VRvaldata3\frames"
    train_set = RGB_dataset(train_folder, process="train", frames_per_data=10)
    train_loader = data.DataLoader(train_set, batch_size=10,shuffle=True,drop_last=True)
    #train_loader=0
    validation_set = RGB_dataset(val_folder,process="train",frames_per_data=10)
    validation_loader = data.DataLoader(validation_set, batch_size=10,drop_last=True)

    optimizer = torch.optim.Adam(att_model.parameters(),lr=1e-5,weight_decay=1e-5)
    criterion = LOSS()


    train(train_loader,validation_loader,optimizer,criterion,model,device,epochs=30,saved_model_path="weights")



