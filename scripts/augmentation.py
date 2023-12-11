import cv2
import numpy as np
import os
import argparse
current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
grant_parent_directory = os.path.dirname(parent_directory)

def augment(path,frame_files_sorted,stored_path,slid):
    count=0

    for r, frame in enumerate(frame_files_sorted):

        if frame.endswith(".png") or frame.endswith('.jpg'):

            a = f"{count:04d}.png"
            print(stored_path + "/" + a)
            img = cv2.imread(path + "/" + frame)
            img = cv2.resize(img, (640, 320))
            cv2.imwrite(stored_path + "/"+a, img)
            fl = cv2.flip(img, 1)
            count += 1
            a = f"{count:04d}.png"

            cv2.imwrite(stored_path + "/" + a, fl)

            step = int(img.shape[1] / slid)

            high = int(img.shape[0] * 0.04)

            upper = img[:high, :, :]
            upper = cv2.resize(upper, (int(img.shape[1]), int(img.shape[0] * 0.06)))
            midle = img[high:-high, :, :]
            down = img[-high:, :, :]
            down = cv2.resize(down, (int(img.shape[1]), int(img.shape[0] * 0.02)))
            im = np.concatenate((upper, midle, down), axis=0)
            im = cv2.resize(im, (int(img.shape[1]), int(img.shape[0])))
            count += 1
            a = f"{count:04d}.png"
            cv2.imwrite(stored_path + "/" + a, im)

            fl = cv2.flip(im, 1)
            count += 1
            a = f"{count:04d}.png"
            cv2.imwrite(stored_path + "/" + a, fl)

            upper = img[:high, :, :]
            upper = cv2.resize(upper, (int(img.shape[1]), int(img.shape[0] * 0.02)))
            midle = img[high:-high, :, :]
            down = img[-high:, :, :]
            down = cv2.resize(down, (int(img.shape[1]), int(img.shape[0] * 0.06)))
            im = np.concatenate((upper, midle, down), axis=0)
            im = cv2.resize(im, (int(img.shape[1]), int(img.shape[0])))
            count += 1
            a = f"{count:04d}.png"
            cv2.imwrite(stored_path + "/" + a, im)

            fl = cv2.flip(im, 1)
            count += 1
            a = f"{count:04d}.png"
            cv2.imwrite(stored_path + "/" + a, fl)

            for i in range(step, 8 * step, step):
                first_part = img[:, :i, :]
                second_part = img[:, i:, :]
                full = np.concatenate((second_part, first_part), axis=1)
                full2 = cv2.flip(full, 1)
                count += 1
                a = f"{count:04d}.png"
                cv2.imwrite(stored_path + "/" + a, full)

                count += 1
                a = f"{count:04d}.png"
                cv2.imwrite(stored_path + "/" + a, full2)

                count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video.')
    parser.add_argument('--path_to_images', type=str,default=r"data/sitzman-salient360",
                        help='path to Sitzamn and Salient360! images')
    parser.add_argument('--path_to_save_augmented_images', type=str, default=r'data/sitzman-salient360',
                        help='Path to save the augmented images')

    args = parser.parse_args()


    path_to_images = args.path_to_images
    output_folder = args.path_to_save_augmented_images
    output_folder = os.path.join(grant_parent_directory, f"{output_folder}")
    if os.path.exists(output_folder):
        print(output_folder + " exists")
    else:
        os.mkdir(output_folder)

    slid = 8
    frames= os.listdir(path_to_images)

    frame_files_sorted = sorted(os.listdir(), key=lambda x: (x.split(".")[0]))

    #frame_files_sorted = [img for img in frame_files_sorted if (img.endswith('.jpg') or img.endswith('.png'))]

    augment(path_to_images,frames,output_folder,slid=8)
