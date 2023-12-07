import cv2
import os
import sys
import ffmpeg

def ERP_videos_to_frames(input_videos_path,output_folder_path):

    videos = os.listdir(input_videos_path)
    for video in videos:

        # Read the video from specified path if your videos are in other format change extention .mp4
        print(video)

        video_ = cv2.VideoCapture(os.path.join(input_videos_path, video))
        base_name, extension = os.path.splitext(video)
        folder_path = output_folder_path+"/"+base_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        fps = video_.get(cv2.CAP_PROP_FPS)
        if fps>45:
            step = round(fps / 30)
            count = 0
            while (True):

                # reading from frame
                ret, frame = video_.read()
                if count % step == 0 and ret:

                    name = os.path.join(folder_path,  f"{count:04d}.png")
                    # writing the extracted images
                    cv2.imwrite(name, frame)


                else:
                    break
                count += 1

            # Release all space and windows once done
            video_.release()
            cv2.destroyAllWindows()
        else:
            count = 0
            while (True):

                # reading from frame
                ret, frame = video_.read()
                if count % step == 0 and ret:

                    name = os.path.join(folder_path, f"{count:04d}.png")
                    # writing the extracted images
                    cv2.imwrite(name, frame)


                else:
                    break
                count += 1

            # Release all space and windows once done
            video_.release()
            cv2.destroyAllWindows()
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python 360toERPframes.py {'360 or erp'} {'input_videos_path'} {'output_folder'}")
        sys.exit(1)
    input_video_type = sys.argv[1]
    input_video_path = sys.argv[2]
    output_folder = sys.argv[3]
    if input_video_type=="erp":
        ERP_videos_to_frames(input_video_path, output_folder)
    else:
        videos360 = os.listdir(input_video_path)
        for video in videos360:
            input_360_video = input_video_path +"/"+video
            output_path = output_folder+"/"+"erp_videos"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_ERP_video = output_path+"/"+video
            try:
                ffmpeg.input(input_360_video).output(output_ERP_video, vf="v360=eac:equirect").run(overwrite_output=True,
                                                                                                 capture_stderr=True)
            except ffmpeg.Error as e:
                print('Error:', e.stderr.decode())
        ERP_videos_to_frames(output_path,output_folder)

