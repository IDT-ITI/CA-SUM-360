import cv2
import numpy as np

import os
import re


def videoCreator(folder_path,video,final_resolution):
    fps=30
    def extract_numerical_part(folder_name):
        if isinstance(folder_name, str):
            match = re.search(r'\d+', folder_name)
            if match:
                return int(match.group())
        return float('inf')

    width = int(final_resolution[1] * 75 / 360)
    height = int(width * 3 / 4)

    output_path =f"2Dvideo_{video}.mp4"

    frames = os.listdir(folder_path)
    frames = sorted(frames, key=extract_numerical_part)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frames:

        current_frame = cv2.imread(os.path.join(folder_path, frame_file))
        video_writer.write(current_frame)

    cv2.destroyAllWindows()

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
    def __init__(self, img_name,final_resolution):

        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._img = cv2.resize(self._img,(final_resolution[0],final_resolution[1]))

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

def fov_extractor(centers,img_path,final_resolution,output_path,video):
    scores_txt = f"saliency_scores_{video}.txt"
    counters=0
    scores = []

    for k, items in enumerate(centers):
        fov = []

        width = int(final_resolution[1] * 75 / 360) # 75 angle for the fov
        height = int(width * 3 / 4)

        for j, item in enumerate(items):
            equ = Equirectangular(img_path[k][j],final_resolution)  # Load equirectangular image
            path2 = img_path[k][j].replace('frames', 'saliency') # load erp saliecy map to extract the saliency scores
            equ2 = Equirectangular(path2, final_resolution)

            x = item[0]
            y = item[1]
            y = final_resolution[0] - y

            longitude = (x / final_resolution[1]) * 360 - 180
            latitude = (y / final_resolution[0]) * 180 - 90

            img = equ.GetPerspective(75, longitude, latitude, height,width)  # Specify parameters(FOV, theta, phi, height, width)

            img2 = equ2.GetPerspective(75, longitude, latitude, height/2, width/2)

            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            saliency_map_filtered = img2.copy()
            saliency_map_filtered[img2 <= 150] = 0
            contours, _ = cv2.findContours(saliency_map_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scores1 = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not np.isnan(img2[y:y + h, x:x + w].mean()):
                    scores1.append(img2[y:y + h, x:x + w].mean())
                else:
                    scores1.append(0.05)
            if not np.isnan(np.mean(scores1)):
                scores.append(np.mean(scores1) / 255)
            else:
                scores.append(0.05)
            # scoress.append(np.mean(img2)/255)



            fov.append(img)
        for r, frame in enumerate(fov):
            transition_frame = frame

            if not os.path.exists(output_path):
                # If it doesn't exist, create it
                os.makedirs(output_path)
            cv2.imwrite(output_path + '/' + f'{counters:06d}.png', frame)
            counters += 1
        with open(scores_txt, "w") as file:
            count1 = 0
            for i, frame in enumerate(scores):
                file.write(f"{frame}\n")
                count1 += 1


def smooth_centers_transition(bbox_frame1, bbox_frame2,cc,resolution):

    interpolation_factor = 0.1
    interpolation_factor1 = 0.02

    #if (abs(2048-max(bbox_frame1[0],bbox_frame2[0])+min(bbox_frame1[0],bbox_frame2[0]))>=400):
    if (abs(bbox_frame1[0]-bbox_frame2[0])<resolution[1]/2):

        x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0])
        y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
        w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
        h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])

    else:
        if max(bbox_frame1[0],bbox_frame2[0]) == bbox_frame1[0]:
            if bbox_frame2[0]>resolution[1]:
                bbox_frame2[0] = resolution[1]-bbox_frame2[0]
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])
            else:
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] + resolution[1]- bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])

        else:
            if bbox_frame1[0]<0:
                bbox_frame1[0] = resolution[1]+bbox_frame1[0]
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0]- bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])
            else:
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - resolution[1]- bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])

    cc+=1
    if x>resolution[1]:
        x = x-resolution[1]
    if x<0:
        x=resolution[1]-x
    center_x = (x+x+w)/2

    center_y = (y+y+h)/2

    if center_x>resolution[1]:
        center_x = center_x - resolution[1]

    return center_x,center_y,x,y,w,h
def final_smooth_centers_transition(bbox_frame1, bbox_frame2,cc,final_resolution):
    if abs(bbox_frame1[0]-bbox_frame2[0])<final_resolution[1]/2 and abs(bbox_frame1[0]-bbox_frame2[0])>int(final_resolution[1]/15):

        interpolation_factor = 0.05
        interpolation_factor1 = 0.025
    else:
        interpolation_factor = 0.03
        interpolation_factor1 = 0.02

    if (abs(bbox_frame1[0]-bbox_frame2[0])<final_resolution[1]/2):

        x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0])
        y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
        w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
        h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])

    else:

        if max(bbox_frame1[0],bbox_frame2[0]) == bbox_frame1[0]:
            if bbox_frame2[0]>final_resolution[1]:
                bbox_frame2[0] = final_resolution[1]-bbox_frame2[0]
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] - bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])
            else:
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] + final_resolution[1]- bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])

        else:
            if bbox_frame1[0]<0:
                bbox_frame1[0] = final_resolution[1]+bbox_frame1[0]
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0]- bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])
            else:
                x = bbox_frame1[0] + interpolation_factor * (bbox_frame2[0] -final_resolution[1]- bbox_frame1[0])
                y = bbox_frame1[1] + interpolation_factor1 * (bbox_frame2[1] - bbox_frame1[1])
                w = bbox_frame1[2] + interpolation_factor * (bbox_frame2[2] - bbox_frame1[2])
                h = bbox_frame1[3] + interpolation_factor1 * (bbox_frame2[3] - bbox_frame1[3])

    if x<0:
        x = final_resolution[1]+x
    cc+=1
    center_x = (x+x+w)/2

    center_y = (y+y+h)/2

    if center_x>final_resolution[1]:
        #print("centerx",center_x)
        center_x = center_x - final_resolution[1]

    return center_x,center_y,x,y,w,h

center = [0, 1]

def extract_2d_videos(lists_of_results,lists_frames,frames_path,output_path,resolution,video):

    distances = []
    list_centers = []
    frame_paths = []
    centers_past = []
    frames1=[]
    for i,bounding_boxes in enumerate(lists_of_results):
        count =0
        cc = 0
        centers1 = []
        paths = []


        for j,(x,y,w,h) in enumerate(bounding_boxes):
            center_x1 = (x + x + w) / 2
            center_y1 = (y + y + h) / 2
            if center_x1>resolution[1]:
                center_x1 = resolution[1]-center_x1
                distance = np.sqrt((center_x1**2)+(center_y1**2))
            else:
                distance = np.sqrt((center_x1 ** 2) + (center_y1 ** 2))
            distances.append(distance)
            if j == 0:
                previous=[x, y, w, h]

                xnew = previous[0]
                ynew = previous[1]
                wnew = previous[2]
                hnew = previous[3]

            else:
                current = [x,y,w,h]
                count+=1
                center_x, center_y, xnew, ynew, wnew, hnew = smooth_centers_transition(previous,current,cc,resolution)
                previous = [xnew,ynew,wnew,hnew]
            centers1.append([xnew,ynew,wnew,hnew])
            img_path = frames_path + "/" + f"{lists_frames[i][j]:04d}.png"
            frames1.append(lists_frames[i][j])
            if i == 0:

                image = cv2.imread(img_path)
                a = image.shape[1]
                b = int(a*0.5)
                final_resolution = (b,a)

                multiplier = final_resolution[1]/resolution[1]
            paths.append(img_path)

        for l,item in enumerate(centers1):
            if (l+3<len(centers1)):
                if (centers1[l][0]>centers1[l+1][0] and centers1[l][0]<centers1[l+2][0]) or (centers1[l][0]<centers1[l+1][0] and centers1[l][0]>centers1[l+2][0]):
                    centers1[l+1][0] = centers1[l][0]
        centers = []


        for l,current in enumerate(centers1):
            center_x = int((current[0]  + current[0]  + current[2])  / 2)
            center_y = int((current[1]  + current[1]  + current[3] ) / 2)
            centers_past.append((center_x,center_y))
        for l,current in enumerate(centers1):
            if l==0:

                center_x = int((current[0]*multiplier+current[0]*multiplier+current[2]*multiplier)/2)
                center_y = int((current[1]*multiplier+current[1]*multiplier+current[3]*multiplier)/2)
                previous = [current[0]*multiplier,current[1]*multiplier,current[2]*multiplier,current[3]*multiplier]

            else:

                current = [current[0]*multiplier,current[1]*multiplier,current[2]*multiplier,current[3]*multiplier]

                center_x, center_y, xnew, ynew, wnew, hnew = final_smooth_centers_transition(previous, current, cc,final_resolution)

                previous = [xnew,ynew,wnew,hnew]

            centers.append([int(center_x),int(center_y)])

        #print("2",centers)
        list_centers.append(centers)
        frame_paths.append(paths)



    fov_extractor(list_centers,frame_paths,final_resolution,output_path,video)




    videoCreator(output_path,video,final_resolution)
