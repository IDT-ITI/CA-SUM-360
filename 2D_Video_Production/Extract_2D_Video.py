import math
from PIL import Image
import cv2
import numpy as np

import os
import re
# Parameters
fps = 30

alpha = 0.00 # Transition blending factor
interpolation_steps = 0


def videoCreator(output_path):
    folder_path = output_path
    list = os.listdir(folder_path)

    def extract_numerical_part(folder_name):
        if isinstance(folder_name, str):
            match = re.search(r'\d+', folder_name)
            if match:
                return int(match.group())
        return float('inf')

    list = sorted(list, key=extract_numerical_part)
    # print(list)
    # Output video file path
    output_path = "2Dvideo.mp4"
    frames = []
    for item in list:

        frame_files = os.listdir(folder_path + "/" + item)

        frames.append(frame_files)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (480, 320))

    for k, frame in enumerate(frames):

        previous_frame = None

        for frame_file in frame:

            current_frame = cv2.imread(os.path.join(folder_path + "/" + f"Out{k}", frame_file))
            current_frame = cv2.resize(current_frame, (480, 320))

            if previous_frame is not None:

                # Filter and interpolate intermediate frames
                for i in range(1, interpolation_steps + 1):
                    alpha_interpolation = i / (interpolation_steps + 1)
                    interpolated_frame = cv2.addWeighted(previous_frame, 1 - alpha_interpolation, current_frame,
                                                         alpha_interpolation, 0)

                    # Write the interpolated frame to the video file
                    video_writer.write(interpolated_frame)

                # Write the current frame to the video file
            video_writer.write(current_frame)

            previous_frame = current_frame
        previous_frame = None
        # Release the video writer and destroy any remaining windows
    video_writer.release()
    cv2.destroyAllWindows()


def smooth_transition(prev_frame, current_frame, next_frame, alpha):

    prev_frame = prev_frame.astype(np.float32)
    current_frame = current_frame.astype(np.float32)
    next_frame = next_frame.astype(np.float32)


    interpolated_frame = (1 - alpha) * (current_frame - prev_frame) + alpha * (next_frame - current_frame)
    interpolated_frame = prev_frame + interpolated_frame


    interpolated_frame = np.clip(interpolated_frame, 0, 255).astype(np.uint8)

    return interpolated_frame
def smooth_bounding_box_transition(bbox_frame1, bbox_frame2,cc,resolution):

    interpolation_factor = 0.03



    #if (abs(2048-max(bbox_frame1[0],bbox_frame2[0])+min(bbox_frame1[0],bbox_frame2[0]))>=400):
    if (abs(bbox_frame1[0]-bbox_frame2[0])<1000):

        x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0]))
        y = int(bbox_frame1[1] + interpolation_factor * (bbox_frame2[1] - bbox_frame1[1]))
        w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
        h = int(bbox_frame1[3] + interpolation_factor * (bbox_frame2[3] - bbox_frame1[3]))

    else:
        if max(bbox_frame1[0],bbox_frame2[0]) == bbox_frame1[0]:
            if bbox_frame2[0]>resolution[1]:
                bbox_frame2[0] = resolution[1]-bbox_frame2[0]
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor * (bbox_frame2[3] - bbox_frame1[3]))
            else:
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] + 2048 - bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor * (bbox_frame2[3] - bbox_frame1[3]))

        else:
            if bbox_frame1[0]<0:
                bbox_frame1[0] = resolution[1]+bbox_frame1[0]
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0]- bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor * (bbox_frame2[3] - bbox_frame1[3]))
            else:
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - resolution[1]- bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor * (bbox_frame2[3] - bbox_frame1[3]))

    cc+=1


    center_x = int((x+x+w)/2)

    center_y = int((y+y+h)/2)
    if center_x>resolution[1]:
        center_x = center_x - resolution[1]
    r = center_y+fov_y2
    if r>resolution[0]:
        center_y = resolution[0]-fov_y2
    return center_x,center_y,x,y,w,h



fov_radius = 240
fov_y1 = 160
fov_y2 = 160
center = [0, 1]

count = 0
fov = []
alpha = 0.2
alpha1 = 0.05

def extract_2d_videos(lists_of_results,lists_frames,scores,frames_path,output_path,resolution):



    file_path = "Subshots_video_.txt"
    file_path1 = "scores_video_.txt"
    c = 0
    transition_frames = []
    for i,bounding_boxes in enumerate(lists_of_results):

        count =0
        cc = 0
        fov = []

        for j,(x,y,w,h) in enumerate(bounding_boxes):
            a=1

            frame1 = cv2.imread(frames_path + "/" + f"{lists_frames[i][j]:04d}.png")  # Load the frame


            frame1 = cv2.resize(frame1, (resolution[1],resolution[0]))
            img = frame1


            if c == 0:
                previous=[x, y, w, h]

                fov_min_row = int((x+x+w)/2 - fov_radius)
                fov_max_row = int((x+x+w)/2 + fov_radius)
                fov_min_col = int((y+y+h)/2 - fov_y1)
                fov_max_col = int((y+y+h)/2 + fov_y2)
                if fov_min_row < 0:

                    extra_cols = -fov_min_row

                    fov_min_row = img.shape[1] - extra_cols  # 2048 - extra rows


                    # Extract the field of view from the image
                    fov_left = img[fov_min_col:fov_max_col, fov_min_row:img.shape[1]]
                    fov_right = img[fov_min_col:fov_max_col, 0:fov_max_row]

                    # Combine the FOV from the left and right sides
                    fov_new = np.concatenate((fov_left, fov_right), axis=1)
                elif fov_max_row > resolution[1]:
                    extra_cols = fov_max_row -resolution[1]

                    fov_max_left = extra_cols
                    fov_right = img[fov_min_col:fov_max_col, 0:fov_max_left]
                    fov_left = img[fov_min_col:fov_max_col, fov_min_row:resolution[1]]
                    fov_new = np.concatenate((fov_left, fov_right), axis=1)
                else:

                    fov_new = img[fov_min_col:fov_max_col, fov_min_row:fov_max_row]

                fov_previous = fov_new


            elif c!=0:



                current = [x,y,w,h]

                count+=1


                center[0], center[1], xnew, ynew, wnew, hnew = smooth_bounding_box_transition(previous,current,cc,resolution)

                previous = [xnew, ynew, wnew, hnew]
                fov_min_row = int(center[0] - fov_radius)
                fov_max_row = int( center[0] + fov_radius)
                fov_min_col = int( center[1] - fov_y1)
                fov_max_col = int(center[1] + fov_y2)



                


                #Case when the FOV extends beyond the left-right boundary
                if fov_min_row < 0:


                    extra_cols = -fov_min_row

                    # Adjust the minimum and maximum column values
                    fov_min_row = img.shape[1]-extra_cols #2048 - extra rows


                    # Extract the field of view from the image
                    fov_left = img[fov_min_col:fov_max_col, fov_min_row:img.shape[1]]
                    fov_right = img[fov_min_col:fov_max_col, 0:fov_max_row]

                    # Combine the FOV from the left and right sides
                    fov_new = np.concatenate((fov_left, fov_right), axis=1)
                elif fov_max_row>resolution[1]:

                    extra_cols = fov_max_row-resolution[1]

                    fov_max_left = extra_cols
                    fov_right = img[fov_min_col:fov_max_col,0:fov_max_left]
                    fov_left = img[fov_min_col:fov_max_col,fov_min_row:resolution[1]]
                    fov_new = np.concatenate((fov_left, fov_right), axis=1)
                else:

                    fov_new = img[fov_min_col:fov_max_col,fov_min_row:fov_max_row]


                count=0
            else:




                center[0], center[1], xnew, ynew, wnew, hnew = smooth_bounding_box_transition(previous,current,cc,resolution)

                previous = [x, y, w, h]

                fov_min_row = int(center[0] - fov_radius)
                fov_max_row = int(center[0] + fov_radius)
                fov_min_col = int(center[1] - fov_y1)
                fov_max_col = int(center[1] + fov_y2)


                # Handle the case when the FOV extends beyond the left-right boundary
                if fov_min_row < 0:
                    # Calculate the additional columns on the right side
                    extra_cols = -fov_min_row


                    fov_min_row = img.shape[1] - extra_cols  # 2048 - extra rows



                    fov_left = img[fov_min_col:fov_max_col, fov_min_row:img.shape[1]]
                    fov_right = img[fov_min_col:fov_max_col, 0:fov_max_row]

                    # Combine the FOV from the left and right sides
                    fov_new = np.concatenate((fov_left, fov_right), axis=1)
                elif fov_max_row > resolution[1]:
                    extra_cols = fov_max_row -resolution[1]
                    fov_min_left = 0
                    fov_max_left = extra_cols
                    fov_right = img[fov_min_col:fov_max_col, 0:fov_max_left]
                    fov_left = img[fov_min_col:fov_max_col, fov_min_row:resolution[1]]
                    fov_new = np.concatenate((fov_left, fov_right), axis=1)
                else:
                    # Calculate the top-left and bottom-right coordinates of the field of view
                    fov_new = img[fov_min_col:fov_max_col, fov_min_row:fov_max_row]



            if c>=1:
                #print(fov_new)

                prev_frame = fov_previous.astype(np.float32)

                prev_frame = cv2.resize(prev_frame, (480,320))

                current_frame = fov_new.astype(np.float32)
                current_frame = cv2.resize(current_frame, (480,320))

                interpolated_frame = (1 - alpha) * (current_frame - prev_frame)
                interpolated_frame = prev_frame + interpolated_frame


                interpolated_frame = np.clip(interpolated_frame, 0, 255).astype(np.uint8)

                fov.append(interpolated_frame)
                fov_previous = interpolated_frame

            c+=1

            prev_frame = frame1.copy()
        for k, fvs in enumerate(fov):
            if k + 3 < len(fov):
                prev_frame = fov[k]
                current_frame = fov[k + 1]
                next_frame = fov[k + 2]
                prev_frame = cv2.resize(prev_frame, (480,320))
                current_frame = cv2.resize(current_frame, (480,320))
                next_frame = cv2.resize(next_frame, (480,320))

                transition_frame = smooth_transition(prev_frame, current_frame, next_frame, alpha1)
                transition_frames.append(transition_frame)

        fov = []

        c = 0
        for r, fr in enumerate(transition_frames):
            path = output_path + "/" + f"Out{i}"
            if not os.path.exists(path):
                os.mkdir(path)


            cv2.imwrite(os.path.join(path, f"{r:04d}.png"), (fr).astype(np.uint8))
                    #cv2.imshow('Smooth Transition', fr)
                    #cv2.waitKey(0)
        transition_frames = []

    count = 0
    with open(file_path, "w") as file:

        for i,frame in enumerate(lists_frames):

            if i==0:
                file.write(f"0000 {(len(frame)-5):04d}\n")
                count = len(frame)-4
            else:
                file.write(f"{(count):04d} {(count+len(frame)-5):04d}\n")
                count+=len(frame)-4
    with open(file_path1, "w") as file:
        count=0
        for i,frame in enumerate(lists_frames):
            count1 = 0
            for j,fr in enumerate(frame):
                if j>3:
                    file.write(f"{scores[i][j]}\n")
                    count+=1
    #y = ccc
    videoCreator(output_path)
