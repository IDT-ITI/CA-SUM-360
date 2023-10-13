import argparse

def args():
    parser = argparse.ArgumentParser(description='Args for 2D video production algorithm.')
    parser.add_argument('--saliency_maps_path', type=str,
                        default=r"D:\Program Files\IoanProjects\summary_data\saliency_sports",
                        help='Directory path of saliency maps')
    parser.add_argument('--frames_path', type=str, default=r"D:\Program Files\IoanProjects\360_Saliency_dataset_2018ECCV",
                        help='Directory path of frames')

    parser.add_argument('--intensity_value', type=int, default=150, help='Intensity value parameter')
    parser.add_argument('--dbscan_distance', type=float, default=600, help='DBSCAN distance parameter')
    parser.add_argument('--spatial_distance', type=int, default=75, help='Spatial distance parameter')
    parser.add_argument('--fill_loss', type=int, default=25, help='Fill loss parameter')

    parser.add_argument('--path_to_folder', type=str,
                        default=r'CA-SUM-360/2D_Video_Production/outputs_fovs',
                        help='Path to save subvideos frames')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1080, 1920], help='Resolution parameter')
    # Your code here
    return parser
if __name__ == '__main__':

    args = args()
    print(args.parse_args())
