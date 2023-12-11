import argparse

def inference_args():
    parser = argparse.ArgumentParser(description='Description for args of inference.py ATSal model')

    # Model parameters
    parser.add_argument('--gpu', type=str, default="cuda:0",
                        help='Use of gpu, if not available chooses "cpu"')
    parser.add_argument('--attention_model', type=str, default="weights/attention.pt",
                        help='Attention model weights path, Default weights/attention.pt')
    parser.add_argument('--expertPoles_model', type=str, default="weights/Poles.pt",
                        help='Expert Poles model weights path, Default weights/Poles.pt')
    parser.add_argument('--expertEquator_model', type=str, default="weights/Equator.pt",
                        help='Expert Equator model weights path, Default weights/Equator.pt')


    #Data loader parameters
    parser.add_argument('--path_to_ERP_frames', type=str,
                        default=r'data/VR-EyeTracking/validation/frames',
                        help='Path to the folder with the extracted ERP frames')
    parser.add_argument('--load_gt', type=str,
                        default='False',
                        help='If you want to calculate metrics define it as True, else False')
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 640], help='Resolution of the ERP images for the model')
    parser.add_argument('--clip_size', type=int, default=10, help='Frames per data for data loader')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')

    #Output_path
    parser.add_argument('--path_to_extracted_saliency_maps', type=str,
                        default=r'outputs',
                        help='Path to the folder for the extracted saliency_maps')

    return parser



if __name__ == '__main__':

   args = inference_args()
   print(args.parse_args())
