import cv2
import numpy as np
import os


def is_camera_moving(frame_folder_path, parameter_1=0.25, resolution=[320,640]):
    frames = [os.path.join(frame_folder_path, frame_name) for frame_name in os.listdir(frame_folder_path) if
              frame_name.lower().endswith(('.png','.jpg'))]
    frames.sort()

    prev = None
    hann = None
    count = 0


    for i, frame_path in enumerate(frames):
        if i > 20 and i + 30 < len(frames):
            #if i>160:
            #print(frame_path)
            if frame_path.lower().endswith(('.png','.jpg')):
                frame = cv2.imread(frame_path)
                frame = cv2.resize(frame, (resolution[1],resolution[0]))
                curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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


                if count == 15:
                    return True

                prev = curr.copy()

    return False



if __name__ == "__main__":

    folder_path = r'D:\Program Files\IoanProjects\VRvaldata3\\frames'
    list_videos = os.listdir(folder_path)
    print(list_videos)
    for item in list_videos:

        path = folder_path + "/" + item


        is_moving = is_camera_moving(path,parameter_1=0.5,resolution=[320,640])

        if is_moving:
            print(item)
            print("The camera is moving.")
