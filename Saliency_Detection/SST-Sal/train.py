import torch
from data_loader import RGB
from LossFunction import KLWeightedLossSequence

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import SST_Sal
import os
from configs import training_args
current_script_path = os.path.abspath(__file__)

# Navigate to the parent directory (one level up)
parent_directory = os.path.dirname(current_script_path)
grant_parent_directory = os.path.dirname(parent_directory)
grant_parent_directory = os.path.dirname(grant_parent_directory)
def load_model(pt_model, new_model):
    temp = torch.load("weights/"+pt_model+'.pth')

    new_model.load_state_dict(temp)
    return new_model

def train_(train_loader, val_loader, model, device, criterion, optimizer, EPOCHS=100,save_model_path="weights"):




    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    model.train()

    model.to(device)
    criterion.cuda(device)



    # Training loop
    for epoch in range(EPOCHS):
        print(f"Training in epoch {epoch}")

        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0

        for k,video in enumerate(train_loader):
            for (x, y) in video:


                optimizer.zero_grad()

                pred = model(x.to(device))

                loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))

                loss.sum().backward()
                optimizer.step()

                avg_loss_train += loss.sum().item()

                counter_train += 1
            if k % 1400 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, k,
                                                                                           len(train_data),
                                                                                           avg_loss_train / counter_train))


        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))



        # Evaluate on validation set
        with torch.no_grad():
            for video in val_loader:

                for (x, y) in video:

                    counter_val += 1
                    pred = model(x.to(device))
                    loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))
                    avg_loss_val += loss.sum().item()
                    batch_size, Nframes, _, _ = pred[:, :, 0, :, :].shape



        print("epoch",epoch)
        print("Train Loss",avg_loss_train / counter_train)
        print("test Loss",avg_loss_val / counter_val)

        scheduler.step(avg_loss_val / counter_val)

        # Save model

        torch.save(model, save_model_path+ '/' + str(epoch) + '_model.pt')








if __name__ == '__main__':

    # Train SST-Sal
    args = training_args().parse_args()
    gpu = args.gpu
    hidden_layers = args.hidden_layers
    path_to_frames_folder = args.path_to_ERP_frames
    process = args.process
    resolution = args.resolution
    clip_size = args.clip_size
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    save_model_path = args.save_model_path
    load_gt = args.load_gt


    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")


    model = SST_Sal(hidden_dim=hidden_layers)

    criterion = KLWeightedLossSequence()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_path_txt = os.path.join(grant_parent_directory,"data/VR-EyeTracking/train_split.txt")
    val_path_txt = os.path.join(grant_parent_directory, "data/VR-EyeTracking/val_split.txt")
    with open(train_path_txt, 'r') as file:
        # Read the content of the file and split it by commas
        content = file.read()
        values = content.split(',')
    static_videos = [value.strip() for value in values]
    
    with open(val_path_txt, 'r') as file:
        content = file.read()
        values = content.split(',')
    static_val_videos = [value.strip() for value in values]
    print(static_val_videos)
    path_to_frames_folder = os.path.join(grant_parent_directory,path_to_frames_folder)
    train_data = RGB(path_to_frames_folder,static_videos,load_gt,process=process,frames_per_data=clip_size,resolution=resolution)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)

    path_to_frames_validation_folder = os.path.join(grant_parent_directory,path_to_frames_folder)
    validation_data = RGB(path_to_frames_folder,static_val_videos,load_gt,process=process,frames_per_data=clip_size,resolution=resolution)
    validation_loader = DataLoader(validation_data, batch_size=1,shuffle=False,drop_last=True)

    model = train_(train_loader, validation_loader, model, device, criterion,optimizer,EPOCHS = epochs,save_model_path=save_model_path)

