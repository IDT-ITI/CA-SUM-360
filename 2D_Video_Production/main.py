import cv2
from Extract_Salient_Regions import extract_salient_regions
from Create_Subvolumes import group_salient_regions
from Extract_2D_Video import extract_2d_videos
import os
from configs import args
current_script_path = os.path.abspath(__file__)

# Navigate to the parent directory (one level up)
parent_directory = os.path.dirname(current_script_path)
grant_parent_directory = os.path.dirname(parent_directory)

def run(**kwargs):
    dbscan_distance = kwargs['dbscan_distance']
    fill_loss = kwargs['fill_loss']
    frames_path = kwargs['frames_folder_path']
    intensity_value = kwargs['intensity_value']
    path_to_folder = kwargs['fov_output_path']
    resolution = kwargs['resolution']
    saliency_maps_path = kwargs['saliency_maps_path']
    spatial_distance = kwargs['spatial_distance']
    frames_path = os.path.join(grant_parent_directory, frames_path)
    saliency_maps_path =  os.path.join(grant_parent_directory, saliency_maps_path)
   
    list_of_videos = os.listdir(frames_path)
    for video in list_of_videos:
        path_to_video_frames = frames_path +"/"+video
        path_to_video_saliency_maps = saliency_maps_path +"/"+video
        list_frames_of_video = os.listdir(path_to_video_frames)
        length = len(list_frames_of_video)
        salient_regions = []
        for i in range(length):  # The total number of frames


            saliency_map = cv2.imread(path_to_video_saliency_maps + "/" + f"{i:04d}.png", cv2.IMREAD_GRAYSCALE)
            saliency_map = cv2.resize(saliency_map, (resolution[1], resolution[0]),interpolation=cv2.INTER_AREA)



            bounding_boxes = extract_salient_regions(saliency_map,intensity_value,dbscan_distance,resolution=[resolution[0],resolution[1]])
            salient_regions.append((f"{i:04d}", bounding_boxes))

        frame_number = [item[0] for item in salient_regions]
        salient_regions_list = [item[1] for item in salient_regions]

        result_lists,frames= group_salient_regions(salient_regions_list,frame_number,path_to_video_saliency_maps,spatial_distance,fill_loss,resolution=[resolution[0],resolution[1]])

        extract_2d_videos(result_lists,frames,path_to_video_frames,output_path=path_to_folder,resolution=resolution,video=video)

if __name__ == '__main__':


    parser = args()
    args = parser.parse_args()

    run(**vars(args))
