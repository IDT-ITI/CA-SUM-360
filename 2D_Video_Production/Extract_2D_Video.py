import cv2
import numpy as np

import os
import re


def videoCreator(output_path):
    folder_path = output_path

    fps=30
    def extract_numerical_part(folder_name):
        if isinstance(folder_name, str):
            match = re.search(r'\d+', folder_name)
            if match:
                return int(match.group())
        return float('inf')


    output_path = f"2Dvideo.mp4"

    frames = os.listdir(folder_path)
    frames = sorted(frames, key=extract_numerical_part)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (480, 360))
    #print(frames)
    for frame_file in frames:

        current_frame = cv2.imread(os.path.join(folder_path, frame_file))
        video_writer.write(current_frame)

    cv2.destroyAllWindows()
def smooth_transition(prev_frame, current_frame, next_frame, alpha):
    prev_frame = prev_frame.astype(np.float32)
    current_frame = current_frame.astype(np.float32)
    next_frame = next_frame.astype(np.float32)


    interpolated_frame = (1 - alpha) * (current_frame - prev_frame) + alpha * (next_frame - current_frame)
    interpolated_frame = prev_frame + interpolated_frame


    interpolated_frame = np.clip(interpolated_frame, 0, 255).astype(np.uint8)

    return interpolated_frame
def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class Equirectangular:
    def __init__(self, img_name):
        #print(img_name)
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._img = cv2.resize(self._img,(1080,1920))

        [self._height, self._width,_] =  self._img.shape


    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T

        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp


def fov_extractor(centers,img_path,resolution,output_path):
    image_height, image_width = resolution

    fovs = []
    for k, items in enumerate(centers):
        fov = []
        for j, item in enumerate(items):
            equ = Equirectangular(img_path[k][j])  # Load equirectangular image

            x = item[0]
            y = item[1]
            y = image_height - y

            longitude = (x / image_width) * 360 - 180
            latitude = (y / image_height) * 180 - 90

            img = equ.GetPerspective(80, longitude, latitude, 360,
                                     480)  # Specify parameters(FOV, theta, phi, height, width)

            fov.append(img)
        fovs.append(fov)
    count = 0
    for r1, frames in enumerate(fovs):
        for r, frame in enumerate(frames):
            if r + 3 < len(frames):

                prev_frame = frames[r]
                current_frame = frames[r + 1]
                next_frame = frames[r + 2]

                transition_frame = smooth_transition(prev_frame, current_frame, next_frame, 0.2)

            else:
                transition_frame = frame
            if not os.path.exists(output_path):
                # If it doesn't exist, create it
                os.makedirs(output_path)
            cv2.imwrite(output_path +"/"+{count:06d}.png', transition_frame)
            count += 1


def smooth_centers_transition(bbox_frame1, bbox_frame2,cc,resolution):

    interpolation_factor = 0.03
    interpolation_factor1 = 0.04

    #if (abs(2048-max(bbox_frame1[0],bbox_frame2[0])+min(bbox_frame1[0],bbox_frame2[0]))>=400):
    if (abs(bbox_frame1[0]-bbox_frame2[0])<1000):

        x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0]))
        y = int(bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1]))
        w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
        h = int(bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3]))

    else:
        if max(bbox_frame1[0],bbox_frame2[0]) == bbox_frame1[0]:
            if bbox_frame2[0]>resolution[1]:
                bbox_frame2[0] = resolution[1]-bbox_frame2[0]
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3]))
            else:
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] + 2048 - bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3]))

        else:
            if bbox_frame1[0]<0:
                bbox_frame1[0] = resolution[1]+bbox_frame1[0]
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0]- bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3]))
            else:
                x = int(bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - resolution[1]- bbox_frame1[0]))
                y = int(bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1]))
                w = int(bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2]))
                h = int(bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3]))

    cc+=1
    center_x = int((x+x+w)/2)

    center_y = int((y+y+h)/2)

    if center_x>resolution[1]:
        center_x = center_x - resolution[1]

    return center_x,center_y,x,y,w,h



center = [0, 1]



def extract_2d_videos(lists_of_results,lists_frames,scores,frames_path,output_path,resolution):
    counter = 0

    file_path = f"Subshots_video_.txt"
    file_path1 = f"scores_video_.txt"
    file_path2 = f"distances_.txt"
    distances = []
    list_centers = []
    frame_paths = []
    for i,bounding_boxes in enumerate(lists_of_results):
        count =0
        cc = 0
        centers = []
        paths = []
        for j,(x,y,w,h) in enumerate(bounding_boxes):
            center_x1 = int((x + x + w) / 2)
            center_y1 = int((y + y + h) / 2)
            if center_x1>resolution[2]-100:
                center_x1 = resolution[2]-center_x1
                distance = np.sqrt((center_x1**2)+(center_y1**2))
            else:
                distance = np.sqrt((center_x1 ** 2) + (center_y1 ** 2))
            distances.append(distance)
            if j == 0:
                previous=[x, y, w, h]
                center_x = int((x+x+w)/2)
                center_y = int((y+y+h)/2)

            else:
                current = [x,y,w,h]
                count+=1
                center_x, center_y, xnew, ynew, wnew, hnew = smooth_centers_transition(previous,current,cc,resolution)
                previous = [xnew,ynew,wnew,hnew]
            centers.append([center_x,center_y])
            img_path = frames_path + "/" + f"{lists_frames[i][j]:04d}.png"
            paths.append(img_path)
        #print("1",centers)
        for l,item in enumerate(centers):
            if (l+3<len(centers)):
                if (centers[l][0]>centers[l+1][0] and centers[l][0]<centers[l+2][0]) or (centers[l][0]<centers[l+1][0] and centers[l][0]>centers[l+2][0]):
                    centers[l+1][0] = centers[l][0]
        #print("2",centers)
        list_centers.append(centers)
        frame_paths.append(paths)



    fov_extractor(list_centers,frame_paths,resolution,output_path)

    count = 0
    with open(file_path, "w") as file:

        for i,frame in enumerate(lists_frames):

            if i==0:
                file.write(f"0000 {(len(frame)-1):04d}\n")
                count = len(frame)
            else:
                file.write(f"{(count):04d} {(count+len(frame)-1):04d}\n")
                count+=len(frame)
    with open(file_path1, "w") as file:
        count1=0
        for i,frame in enumerate(lists_frames):
            for j,fr in enumerate(frame):

                file.write(f"{scores[i][j]}\n")
                count1+=1
    with open(file_path2, "w") as file:
        count2=0
        for i,dist in enumerate(distances):
            file.write(f"{dist}\n")
            count2+=1


    videoCreator(output_path)
