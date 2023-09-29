import cv2
import os
import datetime
import numpy as np
import pickle
import torch
from torch.utils import data
from torchvision import transforms, utils
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from data_loader import Static_dataset
from model import Sal_based_Attention_module
from LossFunction import LOSS
from metrics import Metrics

mean = lambda x: sum(x) / len(x)

def adjust_learning_rate(optimizer,learning_rate, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (decay_rate ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, state='model'):
    # Switch to train mode
    model.train()

    print("Now commencing {} epoch {}".format(state, epoch))

    losses = []
    nss_batch = []
    for j, batch in enumerate(train_loader):


        loss = torch.tensor(0)
        nss_score = 0
        frame, gtruth,fx= batch
        optimizer.zero_grad()
        for i in range(frame.size()[0]):
            # try:

            saliency_map, fiaxtion_module = model(frame[i])
            saliency_map = saliency_map.squeeze(0)
            fiaxtion_module = fiaxtion_module.squeeze(0)


            total_loss, nss = criterion(saliency_map.cpu(), gtruth[i].cpu(),fx[i].cpu(), fiaxtion_module.cpu())
            nss_score = nss_score + nss.item()

            loss = loss.item() + total_loss




        loss.backward()

        optimizer.step()
        nss_score = nss_score / (frame.size()[0])


        losses.append(loss.data / frame.size()[0])
        nss_batch.append(nss_score)
    return (mean(losses), mean(nss_batch))


def validate(val_loader, model, criterion, epoch):
    # Switch to train mode
    model.eval()

    losses = []
    nss_batch = []
    cc_metric = []
    sim_metric = []
    kld_metric = []
    for j, batch in enumerate(val_loader):

        loss = 0
        nss_score = 0
        frame, gtruth,fx = batch
        for i in range(frame.size()[0]):

            with torch.no_grad():
                saliency_map, fiaxtion_module = model(frame[i])
                saliency_map = saliency_map.squeeze(0)
                fiaxtion_module = fiaxtion_module.squeeze(0)

                total_loss, nss = criterion(saliency_map.cpu(), gtruth[i].cpu(), fx[i].cpu(), fiaxtion_module.cpu())
                cc,sim,kld = Metrics1(saliency_map.cpu(),gtruth[i].cpu())
                cc_metric.append(cc)
                sim_metric.append(sim)
                kld_metric.append(kld)
            nss_score = nss_score + nss
            # print("last loss ",last.data)
            # print("attention loss ",attention.data)
            loss = loss + total_loss

            # loss = loss + attention
        nss_score = nss_score / (frame.size()[0])

        losses.append(loss.data / frame.size()[0])
        nss_batch.append(nss_score)
    print("cc metric",np.mean(cc_metric))
    print("sim metric",np.mean(sim_metric))
    print("kld metric",np.mean(kld_metric))
    return (mean(losses), mean(nss_batch))


if __name__ =="__main__":
    model = Sal_based_Attention_module()
    wrihgt = torch.load('weights/initial.pt')
    model.load_state_dict(wrihgt)



    train_set = Static_dataset(root_path=r'C:\Users\ioankont\Documents\Salient360-Sitzman\training',load_gt=True,number_of_frames=1840,resolution=(640, 320),split="train")
    print("Size of train set is {}".format(len(train_set)))
    train_loader = data.DataLoader(train_set, batch_size=80, shuffle=True,drop_last=False)
    valid_set = Static_dataset(root_path=r'C:\Users\ioankont\Documents\Salient360-Sitzman\validation',load_gt=True,number_of_frames=300,resolution=(640, 320),split="validation")
    print("Size of validation set is {}".format(len(valid_set)))
    val_loader = data.DataLoader(valid_set, batch_size=10,  drop_last=False)
    criterion = LOSS()

    optimizer_1 = torch.optim.Adam([
            {'params':model.parameters() , 'lr':1e-5,'weight_decay': 1e-4}])



    cudnn.benchmark = True
    criterion = criterion.cuda()
    # Traning #

    epochs = 60

    model.cuda()


    for epoch in tqdm(range(epochs)):
        train_losses = []
        val_losses = []
        nss_accuracy = []
        nss_validate = []
        print('**** new epoch ****')

        adjust_learning_rate(optimizer_1, 1e-4, epoch, decay_rate=0.1)
        train_loss, nssaccuracy = train(train_loader, model, criterion, optimizer_1, epoch)



        print("Epoch {}/{} done with train loss {} and nss score {}\n".format(epoch, epochs, train_loss, nssaccuracy))


         #   print("Running validation..")
        val_loss, nssvalidate = validate(val_loader, model, criterion, epoch)
        print("Validation loss: {}\t  validation nss {}".format(val_loss, nssvalidate))


        train_losses.append(train_loss.cpu())
        nss_accuracy.append(nssaccuracy)

        val_losses.append(val_loss.cpu())
        nss_validate.append(nssvalidate.cpu())
        print(f"save model epoch {epoch}")


        torch.save(model.state_dict(),f"train_attention_scratch_{epoch}.pt")






    torch.cuda.empty_cache()


