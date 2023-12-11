import cv2
import os
import argparse
import ffmpeg
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
grant_parent_directory = os.path.dirname(parent_directory)

def ERP_videos_to_frames(input_videos_path,output_folder_path):

    videos = os.listdir(input_videos_path)
    for i,video in enumerate(videos):
        if i>1:

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
                    if ret:
                        if count % step == 0 :

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
                    if ret:

                        name = os.path.join(folder_path, f"{count:04d}.png")
                        # writing the extracted images
                        cv2.imwrite(name, frame)

                        count += 1
                    else:
                        break


                # Release all space and windows once done
                video_.release()
                cv2.destroyAllWindows()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video.')
    parser.add_argument('--video_input_type', type=str,help='erp if the input path contains erp videos, 360 if the input path contains 360 videos')
    parser.add_argument('--input_video_path', type=str, default=r'data/output_frames', help='Path to the input videos path')
    parser.add_argument('--output_folder', type=str, help='Folder to save extracted frames')

    args = parser.parse_args()
    input_video_type = args.video_input_type

    input_video_path = args.input_video_path
    output_folder = args.output_folder
    output_folder = os.path.join(grant_parent_directory, f"{output_folder}")
    print(output_folder)
    if os.path.exists(output_folder):
        print(output_folder + "exists")
    else:
        os.mkdir(output_folder)
    if input_video_type=="erp":

        ERP_videos_to_frames(input_video_path, output_folder)
    else:

        videos360= [f for f in os.listdir(input_video_path) if f.endswith('.mp4')]
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


