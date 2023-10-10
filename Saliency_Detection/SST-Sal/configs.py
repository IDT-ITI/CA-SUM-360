import argparse
def training_args():
    parser = argparse.ArgumentParser(description='Description for args of Train.py SST-Sal model')

    # Model parameters
    parser.add_argument('--gpu', type=str, default="cuda:0",
                        help='Use of gpu, if not available chooses "cpu"')
    parser.add_argument('--hidden_layers', type=int,default=9,
                        help='The number of hidden layers of the model')

    #Data loader parameters
    parser.add_argument('--path_to_frames_folder', type=str,
                        default=r'data/VR-EyeTracking/training/frames',
                        help='Path to the folder with the extracted ERP training frames')
    parser.add_argument('--path_to_frames_validation_folder', type=str,
                        default=r'data/VR-EyeTracking/validation/frames',
                        help='Path to the folder with the extracted ERP validation frames')
    parser.add_argument('--process', type=str,
                        default='train',
                        help='Process for data loader for train.py')
    parser.add_argument('--resolution', type=int, nargs=2, default=[240,320], help='Resolution of the ERP images for the model')
    parser.add_argument('--clip_size', type=int, default=20, help='Frames per data for data loader 20 ')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--epochs',type=int, default=200, help='Number of epochs')

    #Optimizer
    parser.add_argument('--lr', type=int, default=1e-3,help='Learning rate for the BCE optimizer')



    #Output_path
    parser.add_argument('--save_model_path', type=str,
                        default=r'weights',
                        help='Path to the folder for the saved model per epoch')

    return parser

def inference_args():
    parser = argparse.ArgumentParser(description='Description for args of inference.py SST-Sal model')

    # Model parameters
    parser.add_argument('--gpu', type=str, default="cuda:0",
                        help='Use of gpu, if not available chooses "cpu"')
    parser.add_argument('--sst_sal', type=str, default="weights/SST_Sal.pth",
                        help='SST-SAL.pth model weights, Default weights/SalEMA.pt')


    # Data loader parameters

    parser.add_argument('--path_to_frames_validation_folder', type=str,
                        default=r'data/VR-EyeTracking/validation/frames',
                        help='Path to the folder with the extracted ERP validation frames')
    parser.add_argument('--process', type=str,
                        default='test',
                        help='Process for data loader for inference.py')
    parser.add_argument('--resolution', type=int, nargs=2, default=[240, 320],
                        help='Resolution of the ERP images for the model')
    parser.add_argument('--clip_size', type=int, default=20,
                        help='Frames per data for data loader ')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')


    # Output_path
    parser.add_argument('--path_to_save_saliency_maps', type=str,
                        default=r'outputs',
                        help='Path to the folder for saving saliency maps')


    return parser



if __name__ == '__main__':

   args = training_args()
   print(args.parse_args())
