import argparse

def args():
    parser = argparse.ArgumentParser(description='Args for 2D video production algorithm.')
    parser.add_argument('--saliency_maps_path', type=str,
                        default=r"CA-SUM-360-main/data/VR-EyeTracking/saliency",
                        help='Directory path of saliency maps')
    parser.add_argument('--frames_path', type=str, default=r"CA-SUM-360-main/data/VR-EyeTracking/frames",
                        help='Directory path of frames')

    parser.add_argument('--intensity_value', type=int, default=150, help='Intensity value parameter')
    parser.add_argument('--dbscan_distance', type=float, default=1.5, help='DBSCAN distance parameter')
    parser.add_argument('--spatial_distance', type=int, default=85, help='Spatial distance parameter')
    parser.add_argument('--fill_loss', type=int, default=60, help='Fill loss parameter')

    parser.add_argument('--path_to_folder', type=str,
                        default=r'outputs_fovs',
                        help='Path to save subvideos frames')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1080, 1920], help='Resolution parameter')
    # Your code here
    return parser
if __name__ == '__main__':

    args = args()
    print(args.parse_args())
