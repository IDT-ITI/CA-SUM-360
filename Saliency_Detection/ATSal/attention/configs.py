import argparse

def training_args():
    parser = argparse.ArgumentParser(description='Description for args of train.py attention model')

    # Model parameters
    parser.add_argument('--gpu', type=str, default="cuda:0",
                        help='Use of gpu, if not available chooses "cpu"')
    parser.add_argument('--attention_model', type=str, default="weights/initial.pt",
                        help='Attention model weights path, Default attention/weights/initial.pt')

    # Data loader parameters
    parser.add_argument('--path_to_ERP_frames', type=str,
                        default=r'data\VR-EyeTracking\erp_frames\frames',
                        help='Path to the folder with the extracted ERP frames')
    parser.add_argument('--process', type=str,
                        default='train',
                        help='Process for data loader for train.py')
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 640],
                        help='Resolution of the ERP images for the model')
    parser.add_argument('--clip_size', type=int, default=1, help='Frames per data for data loader')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for data loader')

    parser.add_argument('--dataset',type=str,
                        default='VR-EyeTracking',
                        help='if your dataset is Vr-EyeTracking or Salient360!-Sitzman')

    #optimizer
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate for ADAM optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight Decay for ADAM optimizer')

    parser.add_argument('--epochs', type=int, default=90, help='90 epochs for train on Salient360-Sitzman dataset,10 for VR-EyeTracking')
    # Output_path
    parser.add_argument('--model_storage_path', type=str,
                        default=r'Saliency_Detection/ATSal/weights',
                        help='Path to save the checkpoint_weights')

    return parser




if __name__ == '__main__':

   args = training_args()
   print(args)
