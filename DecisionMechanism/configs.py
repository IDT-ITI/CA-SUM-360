import argparse
def args():
    parser = argparse.ArgumentParser(description='Description for Decision Mechanism parameters')
    parser.add_argument('--frames_folder_path', type=str,
                        default=r'CA-SUM-360\output_frames',
                        help='Directory path to equirectangular (ERP) frames')
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 640], help='Resolution parameter')


    parser.add_argument('--parameter_1', type=int, default=0.5, help='Parameter for moving threshold')


    return parser
if __name__ == '__main__':

    args = args()