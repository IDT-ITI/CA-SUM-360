import argparse

def args():
    parser = argparse.ArgumentParser(description='Your script description here.')

    parser.add_argument('--frames_path', type=str, default=r"D:\Program Files\IoanProjects\StaticVRval\frames1\180",
                        help='Directory path of frames')
    parser.add_argument('--saliency_maps_path', type=str,
                        default=r"D:\Program Files\IoanProjects\StaticVRval\saliency1\180",
                        help='Directory path of saliency maps')
    parser.add_argument('--intensity_value', type=int, default=150, help='Intensity value parameter')
    parser.add_argument('--dbscan_distance', type=float, default=1.2, help='DBSCAN distance parameter')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1024, 2048], help='Resolution parameter')
    parser.add_argument('--spatial_distance', type=int, default=100, help='Spatial distance parameter')
    parser.add_argument('--fill_loss', type=int, default=100, help='Fill loss parameter')
    parser.add_argument('--path_to_folder', type=str,
                        default=r'C:\Users\ioankont\PycharmProjects\SalientRegionExtractor\summaryFolder',
                        help='Path to save subvideos frames')
    # Your code here
    return parser
if __name__ == '__main__':

   args = args()
