import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
from data_loader import Multiexpert_dataset
from metrics import Metrics
from models import Poles, Equator
from torch import nn
import math
import os
from configs import training_args

mean = lambda x: sum(x) / len(x)
current_script_path = os.path.abspath(__file__)

# Navigate to the parent directory (one level up)
parent_directory = os.path.dirname(current_script_path)
parent_directory = os.path.dirname(parent_directory)
grant_parent_directory = os.path.dirname(parent_directory)
grant_parent_directory = os.path.dirname(grant_parent_directory)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def main(train_loader, val_loader, model, criterion, optimizer, temporal, dtype, saved_model_path, epochs, keyword):
    for epoch in range(epochs):

        best_loss = 100
        train_loss, optimizer = train(train_loader, model, criterion, optimizer, epoch, dtype)

        print("Epoch {}/{} done with train loss {}\n".format(epoch, epochs, train_loss))

        print("Running validation..")
        val_loss = validate(val_loader, model, criterion, epoch, temporal, dtype)
        print("Validation loss: {}".format(val_loss))
        if keyword == "Equator":
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model, saved_model_path + f"/{keyword}.pt")
        else:
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model, saved_model_path + f"/{keyword}.pt")

        model = model.cuda()


def train(train_loader, model, criterion, optimizer, epoch, dtype):
    # Switch to train mode
    model.train()

    video_losses = []
    print("Now commencing epoch {}".format(epoch))
    for i, video in enumerate(train_loader):

        accumulated_losses = []

        state = None  # Initially no hidden state
        for j, (clip, gtruths) in enumerate(video):

            optimizer.zero_grad()

            clip = Variable(clip.type(dtype).transpose(0, 1))
            gtruths = Variable(gtruths.type(dtype).transpose(0, 1))

            # print(clip.size()) #works! torch.Size([5, 1, 1, 360, 640])
            loss = 0
            for idx in range(clip.size()[0]):
                state, saliency_map = model.forward(input_=clip[idx],
                                                    prev_state=state)  # Based on the number of epoch the model will unfreeze deeper layers moving on to shallow ones

                saliency_map = saliency_map.squeeze(0)

                loss = loss + criterion(saliency_map, gtruths[idx])

            # Keep score
            accumulated_losses.append(loss.data)

            # Compute gradient
            loss.backward()

            # Clip gradient to avoid explosive gradients. Gradients are accumulated so I went for a threshold that depends on clip length. Note that the loss that is stored in the score for printing does not include this clipping.
            nn.utils.clip_grad_norm_(model.parameters(), 10 * clip.size()[0])

            optimizer.step()

            state = repackage_hidden(state)

        video_losses.append(mean(accumulated_losses))
    print(f'Epoch: {epoch} Training Loss: {mean(accumulated_losses)} ')
    return (mean(video_losses), optimizer)


def validate(val_loader, model, criterion, epoch, temporal, dtype):
    # switch to evaluate mode
    model.eval()

    video_losses = []
    cc_metric = []
    sim_metric = []

    with torch.no_grad():
        for i, video in enumerate(val_loader):
            accumulated_losses = []
            state = None  # Initially no hidden state
            for j, (clip, gtruths) in enumerate(video):

                clip = Variable(clip.type(dtype).transpose(0, 1))
                gtruths = Variable(gtruths.type(dtype).transpose(0, 1))

                loss = 0
                for idx in range(clip.size()[0]):

                    if temporal:
                        state, saliency_map = model.forward(clip[idx], state)
                    else:
                        saliency_map = model.forward(clip[idx])

                    saliency_map = saliency_map.squeeze(0)

                    # Compute loss
                    loss = loss + criterion(saliency_map, gtruths[idx])
                    cc, sim = Metrics(saliency_map[0].cpu(), gtruths[idx][0].cpu())
                    if math.isnan(cc):
                        a = 0
                    else:
                        cc_metric.append(cc)
                    if math.isnan(sim):
                        a = 0
                    else:
                        sim_metric.append(sim)
                    # sim_metric.append(sim)
                    # kld_metric.append(kld)
                if temporal:
                    state = repackage_hidden(state)

                # Keep score
                accumulated_losses.append(loss.data)

            video_losses.append(mean(accumulated_losses))
    print(np.mean(cc_metric))
    print(np.mean(sim_metric))
    # print(np.mean(kld_metric))

    return (mean(video_losses))


if __name__ == '__main__':
    args = training_args().parse_args()
    # Equator takes x0,x1,x2,x3 faces of cube map while Poles takes x4,x5 north and south as input, in this example we load Poles
    gpu = args.gpu
    expert_model = args.expert_model
    alpha_parameter = args.alpha_parameter
    ema_loc = args.ema_loc

    lr = args.lr
    weight_decay = args.weight_decay
    path_to_frames_folder = args.path_to_training_cmp_frames
    path_to_frames_validation_folder = args.path_to_validation_cmp_frames
    process = args.process
    resolution = args.resolution
    clip_size = args.clip_size
    batch_size = args.batch_size
    save_model_path = args.model_storage_path

    epochs = args.epochs

    LEARN_ALPHA_ONLY = False

    print("Commencing training on dataset")
    # path_to_frames_folder = os.path.join(grant_parent_directory,path_to_frames_folder)
    path_to_frames_folder = os.path.join(grant_parent_directory, path_to_frames_folder)
    train_videos = os.listdir(path_to_frames_folder)
    train_set = Multiexpert_dataset(root_path=path_to_frames_folder, video_names=train_videos, process=process,
                                    frames_per_data=clip_size, resolution=resolution)
    print("Size of train set is {}".format(len(train_set)))
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    path_to_frames_validation_folder = os.path.join(grant_parent_directory,path_to_frames_validation_folder)
    validation_videos = os.listdir(path_to_frames_validation_folder)
    val_set = Multiexpert_dataset(root_path=path_to_frames_validation_folder, video_names=validation_videos,
                                  process=process, frames_per_data=clip_size, resolution=resolution)
    print("Size of validation set is {}".format(len(val_set)))
    val_loader = data.DataLoader(val_set, batch_size=batch_size, drop_last=True)

    temporal = True

    model = torch.load("salEMA30.pt")
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam([
        {'params': model.salgan.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.alpha, 'lr': 0.1}])

    assert torch.cuda.is_available(), \
        "CUDA is not available in your machine"

    model = model.cuda()
    dtype = torch.cuda.FloatTensor

    criterion = criterion.cuda()
    train_losses = []
    val_losses = []

    save_model_path = os.path.join(grant_parent_directory, save_model_path)
    keyword = "poles"
    if keyword in path_to_frames_folder:
        keyword = "Poles"
    else:
        keyword = "Equator"
    main(train_loader, val_loader, model, criterion, optimizer, temporal, dtype, save_model_path, epochs, keyword)
