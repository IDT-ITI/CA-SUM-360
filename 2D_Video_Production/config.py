import argparse

def main(args):
    frames_path = args.frames_path
    saliency_maps_path = args.saliency_maps_path
    intensity_value = args.intensity_value
    dbscan_distance = args.dbscan_distance
    resolution = args.resolution
    spatial_distance = args.spatial_distance
    fill_loss = args.fill_loss
    path_to_folder = args.path_to_folder

    # Your code here

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define the parameters for the 2D Video Production')

    parser.add_argument('--frames_path', type=str, default=r"D:\Program Files\IoanProjects\VR-EyeTracking\frames\180", help='Directory path of frames')
    parser.add_argument('--saliency_maps_path', type=str, default=r"D:\Program Files\IoanProjects\VR-EyeTracking\saliency\180", help='Directory path of saliency maps')
    parser.add_argument('--intensity_value', type=int, default=150, help='Intensity value parameter')
    parser.add_argument('--dbscan_distance', type=float, default=1.2, help='DBSCAN distance parameter')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1024, 2048], help='Resolution parameter')
    parser.add_argument('--spatial_distance', type=int, default=100, help='Spatial distance parameter')
    parser.add_argument('--fill_loss', type=int, default=100, help='Fill loss parameter')
    parser.add_argument('--path_to_folder', type=str, default=r'C:\Users\ioankont\PycharmProjects\SalientRegionExtractor\summaryFolder', help='Path to save subvideos frames')

    args = parser.parse_args()
    main(args)
