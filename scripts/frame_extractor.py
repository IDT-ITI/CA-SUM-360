import cv2
import os

my_path = r'relative_path_to_your_video\video.mp4' #path .mp4 videos
out_path = r'relative_Path\CA-SUM-360\output_frames' #path to save each video's frames
#os.mkdir(out_path)
for vid in os.listdir(my_path):
    # Read the video from specified path if your videos are in other format change extention .mp4
    print(vid)

    cam = cv2.VideoCapture(os.path.join(my_path, vid))
    base_name, extension = os.path.splitext(vid)
    folder_path = out_path+"/"+base_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    currentframe = 0
    names = []
    count = 0
    while (True):

        # reading from frame
        ret, frame = cam.read()
        if ret:

            name = os.path.join(folder_path,  f"{count:04d}.png")
            # writing the extracted images
            cv2.imwrite(name, frame)
            count+=1

        else:
            break

    # Release all space and windows once done
    cam.release()

    cv2.destroyAllWindows()
