import cv2
import numpy as np
import os
from configs import args
current_script_path = os.path.abspath(__file__)

# Navigate to the parent directory (one level up)
parent_directory = os.path.dirname(current_script_path)
grant_parent_directory = os.path.dirname(parent_directory)


def is_camera_moving(**kwargs):
    parameter_1 = kwargs['parameter_1']
    resolution = kwargs['resolution']
    frame_folder_path= kwargs['frames_folder_path']
    frame_folder_path = os.path.join(grant_parent_directory,frame_folder_path)
    list_videos = os.listdir(frame_folder_path)

    for video in list_videos:
        path = frame_folder_path + "/" + video
        frames = os.listdir(path)

        frames.sort()

        prev = None
        hann = None
        count = 0
        moving = False

        for i, frame in enumerate(frames):
            if i > 50 and i + 50 < len(frames): #avoid first 50 frames and last 20 because some videos starts with black screen, something that affects the final result

                if frame.lower().endswith(('.png','.jpg')):
                    frame_path = path +"/"+frame
                    frame_ = cv2.imread(frame_path)
                    frame_ = cv2.resize(frame_, (resolution[1],resolution[0]))
                    curr = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)

                    if prev is None:
                        prev = curr.copy()

                        hann = cv2.createHanningWindow(curr.shape[::-1], cv2.CV_64F)

                    prev64f = prev.astype(np.float64)
                    curr64f = curr.astype(np.float64)

                    shift, _ = cv2.phaseCorrelate(prev64f, curr64f, hann)


                    # Get boxes dimensions
                    box_width, box_height =resolution[1],int(resolution[0]/12)
                    box_width1,box_height1 = int(resolution[1]/12), int(resolution[0]/10)


                    # Calculate the phase correlation for each region on north and on south

                    top_shift, _ = cv2.phaseCorrelate(prev64f[:box_height, :box_width], curr64f[:box_height, :box_width],
                                                      hann[:box_height, :box_width])
                    bottom_shift, _ = cv2.phaseCorrelate(prev64f[resolution[0]-box_height:resolution[0], :box_width],
                                                         curr64f[resolution[0]-box_height:resolution[0], :box_width], hann[resolution[0]-box_height:resolution[0], :box_width])
                    left_shift1, _ = cv2.phaseCorrelate(prev64f[:box_height1, :box_width1], curr64f[:box_height1, :box_width1],
                                                       hann[:box_height1, :box_width1])
                    left_shift2, _ = cv2.phaseCorrelate(prev64f[resolution[0]-box_height1:resolution[0], :box_width1],
                                                       curr64f[resolution[0]-box_height1:resolution[0], :box_width1],
                                                       hann[resolution[0]-box_height1:resolution[0], :box_width1])
                    right_shift1, _ = cv2.phaseCorrelate(prev64f[:box_height1, resolution[1]-box_width1:resolution[1]],
                                                        curr64f[:box_height1, -box_width1:], hann[:box_height1, -box_width1:])
                    right_shift2, _ = cv2.phaseCorrelate(prev64f[resolution[0]-box_height1:resolution[0], resolution[1] - box_width1:resolution[1]],
                                                        curr64f[resolution[0]-box_height1:resolution[0], -box_width1:],
                                                        hann[resolution[0]-box_height1:resolution[0], -box_width1:])

                    # Calculate the Euclidean norm for each box
                    top_radius = np.sqrt(top_shift[0] ** 2 + top_shift[1] ** 2)
                    bottom_radius = np.sqrt(bottom_shift[0] ** 2 + bottom_shift[1] ** 2)
                    left_radius1 = np.sqrt(left_shift1[0] ** 2 + left_shift1[1] ** 2)
                    right_radius1 = np.sqrt(right_shift1[0] ** 2 + right_shift1[1] ** 2)
                    left_radius2 = np.sqrt(left_shift2[0] ** 2 + left_shift2[1] ** 2)
                    right_radius2 = np.sqrt(right_shift2[0] ** 2 + right_shift2[1] ** 2)

                    sum = (top_radius+bottom_radius +left_radius1+right_radius1+left_radius2+right_radius2)/6
                    #check if there is big motion in the specified regions
                    if sum>parameter_1:
                        count+=1


                    if count == 30:

                        moving = True
                        break

                    prev = curr.copy()
        if moving== True:
            print(f"the {video} is moving")
        else:
            print(f"the {video}is static")




if __name__ == "__main__":

    parser = args()
    args = parser.parse_args()


    is_camera_moving(**vars(args))


