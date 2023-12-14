import argparse
def training_args():
    parser = argparse.ArgumentParser(description='Description for args of train expert model')

    # Model parameters
    parser.add_argument('--gpu', type=str, default="cuda:0",
                        help='Use of gpu, if not available chooses "cpu"')

    parser.add_argument('--expert_model', type=str, default="weights/salEMA30.pt",
                        help='Expert Equator/Poles model weights path, Default weights/SalEMA.pt')
    parser.add_argument('--alpha_parameter', type=float, default=0.1,
                        help='Initial value for parameter Alpha of the model')
    parser.add_argument('-ema_loc',type=int, default=30, help='Input number of layer to place EMA on')

    #Data loader parameters
    parser.add_argument('--path_to_training_cmp_frames', type=str,
                        default=r'data/VR-EyeTracking/cmp_frames/equator/training/frames',
                        help='Path to the folder with the extracted CMP training frames')
    parser.add_argument('--path_to_validation_cmp_frames', type=str,
                        default=r'data/VR-EyeTracking/cmp_frames/equator/validation/frames',
                        help='Path to the folder with the extracted CMP validation frames')
    parser.add_argument('--process', type=str,
                        default='train',
                        help='Process for data loader for inference.py')
    parser.add_argument('--resolution', type=int, nargs=2, default=[160,160], help='Resolution of the ERP images for the model')
    parser.add_argument('--clip_size', type=int, default=10, help='Frames per data for data loader 10 for Equator 80 for Poles')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--epochs',type=int, default=10, help='Number of epochs')

    #Optimizer
    parser.add_argument('--lr', type=int, default=1e-6,help='Learning rate for the BCE optimizer')
    parser.add_argument('--weight_decay', type=int, default=1e-4,
                        help='Learning rate for the BCE optimizer')


    #Output_path
    parser.add_argument('--model_storage_path', type=str,
                        default=r'weights',
                        help='Path to the folder for the saved model per epoch')

    return parser



if __name__ == '__main__':

   args = training_args()
   print(args.parse_args())
