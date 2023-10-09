import cv2
from Extract_Salient_Regions import extract_salient_regions
from Create_Subvolumes import group_salient_regions
from Extract_2D_Video import extract_2d_videos
import os
from configs import args

def run(**kwargs):
    dbscan_distance = kwargs['dbscan_distance']
    fill_loss = kwargs['fill_loss']
    frames_path = kwargs['frames_path']
    intensity_value = kwargs['intensity_value']
    path_to_folder = kwargs['path_to_folder']
    resolution = kwargs['resolution']
    saliency_maps_path = kwargs['saliency_maps_path']
    spatial_distance = kwargs['spatial_distance']
    
   
    list_of_saliency_paths = os.listdir(saliency_maps_path)
    length = len(list_of_saliency_paths)
    salient_regions = []
    for i in range(length):  # Replace  with the total number of frames


        saliency_map = cv2.imread(saliency_maps_path + "/" + f"{i:04d}.png", cv2.IMREAD_GRAYSCALE)
        saliency_map = cv2.resize(saliency_map, (resolution[1], resolution[0]),interpolation=cv2.INTER_AREA)



        bounding_boxes,salient_scores = extract_salient_regions(saliency_map,intensity_value,dbscan_distance,resolution=[resolution[0],resolution[1]])
        salient_regions.append((f"{i:04d}", bounding_boxes,salient_scores))

    frame_number = [item[0] for item in salient_regions]
    # Example usage:

    salient_regions_list = [item[1] for item in salient_regions]

    saliency_scores = [item[2] for item in salient_regions]


    result_lists,frames,saliency_scores = group_salient_regions(salient_regions_list,frame_number,saliency_scores,saliency_maps_path,spatial_distance,fill_loss,resolution=[resolution[0],resolution[1]])

    extract_2d_videos(result_lists,frames,saliency_scores,frames_path,output_path=path_to_folder,resolution=resolution)

if __name__ == '__main__':


    parser = args()
    args = parser.parse_args()

    run(**vars(args))
