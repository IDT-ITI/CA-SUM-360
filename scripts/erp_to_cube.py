import os

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
import argparse

current_script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_script_path)
grant_parent_directory = os.path.dirname(parent_directory)
def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(e_img, [coor_y, coor_x], order=order, mode='wrap')[..., 0]


def equirect_uvgrid(h, w):
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi, -np.pi, num=h, dtype=np.float32) / 2

    return np.stack(np.meshgrid(u, v), axis=-1)


def equirect_facetype(h, w):
    '''
    0F 1R 2B 3L 4U 5D
    '''
    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), np.bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)


def cube_h2dict(cube_h):
    cube_list = cube_h2list(cube_h)
    return dict([(k, cube_list[i]) for i, k in enumerate(['F', 'R', 'B', 'L', 'U', 'D'])])


def cube_h2list(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    return np.split(cube_h, 6, axis=1)


def cube_list2h(cube_list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return np.concatenate(cube_list, axis=1)


def cube_dict2h(cube_dict, face_k=['F', 'R', 'B', 'L', 'U', 'D']):
    # assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def xyzcube(face_w):
    '''
    Return the xyz cordinates of the unit cube in [F R B L U D] format.
    '''
    out = np.zeros((face_w, face_w * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(rng, -rng), -1)

    # Front face (z = 0.5)
    out[:, 0 * face_w:1 * face_w, [0, 1]] = grid
    out[:, 0 * face_w:1 * face_w, 2] = 0.5

    # Right face (x = 0.5)
    out[:, 1 * face_w:2 * face_w, [2, 1]] = grid
    out[:, 1 * face_w:2 * face_w, 0] = 0.5

    # Back face (z = -0.5)
    out[:, 2 * face_w:3 * face_w, [0, 1]] = grid
    out[:, 2 * face_w:3 * face_w, 2] = -0.5

    # Left face (x = -0.5)
    out[:, 3 * face_w:4 * face_w, [2, 1]] = grid
    out[:, 3 * face_w:4 * face_w, 0] = -0.5

    # Up face (y = 0.5)
    out[:, 4 * face_w:5 * face_w, [0, 2]] = grid
    out[:, 4 * face_w:5 * face_w, 1] = 0.5

    # Down face (y = -0.5)
    out[:, 5 * face_w:6 * face_w, [0, 2]] = grid
    out[:, 5 * face_w:6 * face_w, 1] = -0.5

    return out


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return np.concatenate([coor_x, coor_y], axis=-1)


def e2c(e_img, face_w=256, mode='bilinear', cube_format='dict'):
    '''
    e_img:  ndarray in shape of [H, W, *]
    face_w: int, the length of each face of the cubemap
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = xyzcube(face_w)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    cubemap = np.stack([
        sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    return cube_h2dict(cubemap)


def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(e_img, [coor_y, coor_x],
                           order=order, mode='wrap')[..., 0]


def sample_cubefaces(cube_faces, tp, coor_y, coor_x, order):
    cube_faces = cube_faces.copy()
    cube_faces[1] = np.flip(cube_faces[1], 1)
    cube_faces[2] = np.flip(cube_faces[2], 1)
    cube_faces[4] = np.flip(cube_faces[4], 0)

    # Pad up down
    pad_ud = np.zeros((6, 2, cube_faces.shape[2]))
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], 1)

    # Pad left right
    pad_lr = np.zeros((6, cube_faces.shape[1], 2))
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], 2)

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode='wrap')


def c2e(cubemap, h, w, mode='bilinear', cube_format='dict'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec

out_state = {'F':None, 'R':None, 'B':None, 'L':None, 'U':None, 'D':None}





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='erp_to_cube')
    parser.add_argument('--path_to_erp_video_frames',type=str,default=r"data/VR-EyeTracking/erp_frames/frames", help='Path to the folder that contains the video folders in erp frames')
    parser.add_argument('--equator_save_path', type=str,default="data/VR-EyeTracking/cmp_frames/equator/training/frames",
                        help='Equator path')
    parser.add_argument('--poles_save_path', type=str, default=r'data/VR-EyeTracking/cmp_frames/poles/training/frames',
                        help='Path to the input videos path')



    args = parser.parse_args()
    path_to_erp_videos_frames= args.path_to_erp_video_frames

    equator_save_path = args.equator_save_path
    equator_save_path = os.path.join(grant_parent_directory, f"{equator_save_path}")
    poles_save_path = args.poles_save_path
    poles_save_path = os.path.join(grant_parent_directory, f"{poles_save_path}")
    if os.path.exists(equator_save_path):
        print(equator_save_path + "exists")
    else:
        os.mkdir(equator_save_path)
    if os.path.exists(poles_save_path):
        print(poles_save_path + "exists")
    else:
        os.mkdir(poles_save_path)

    train_path_txt = os.path.join(grant_parent_directory, "data/VR-EyeTracking/training_data_split.txt")
    val_path_txt = os.path.join(grant_parent_directory, "data/VR-EyeTracking/validation_data_split.txt")
    with open(train_path_txt, 'r') as file:
        # Read the content of the file and split it by commas
        content = file.read()
        values = content.split(',')
    train_videos = [value.strip() for value in values]

    with open(val_path_txt, 'r') as file:
        content = file.read()
        values = content.split(',')
    validation_videos = [value.strip() for value in values]

    list_of_videos = os.listdir(os.path.join(grant_parent_directory, path_to_erp_videos_frames))
    # Replace 'folder_path' with the path to the folder you want to empty
    length = len(list_of_videos)



    equator_save_validation_path = equator_save_path.replace("training","validation")
    poles_save_validation_path = poles_save_path.replace("training", "validation")

    if os.path.exists(equator_save_validation_path):
        print(equator_save_path + "exists")
    else:
        os.mkdir(equator_save_validation_path)
    if os.path.exists(poles_save_validation_path):
        print(poles_save_path + "exists")
    else:
        os.mkdir(poles_save_validation_path)

    if os.path.exists(equator_save_path + f"/0"):
        print("exists")
    else:
        for r,video in enumerate(train_videos):
            os.mkdir(equator_save_path + f"/{r}")
            os.mkdir(equator_save_path + f"/{r+length}")
            os.mkdir(equator_save_path + f"/{r+2*length}")
            os.mkdir(equator_save_path + f"/{r+3*length}")
            os.mkdir(poles_save_path + f"/{r}")
            os.mkdir(poles_save_path + f"/{r + length}")

        for r,video in enumerate(validation_videos):
            os.mkdir(equator_save_validation_path + f"/{r}")
            os.mkdir(equator_save_validation_path+ f"/{r + length}")
            os.mkdir(equator_save_validation_path + f"/{r + 2 * length}")
            os.mkdir(equator_save_validation_path + f"/{r + 3 * length}")
            os.mkdir(poles_save_validation_path + f"/{r}")
            os.mkdir(poles_save_validation_path + f"/{r + length}")

    for r,video in enumerate(train_videos):
        path1 = path_to_erp_videos_frames +"/"+video
        list = os.listdir(path1)
        for j,item in enumerate(list):
            img = os.path.join(path1,item)
            img = cv2.imread(img)
            img = np.array(img)
            out = e2c(img, face_w=160 , mode='bilinear', cube_format='dict')
            out_predict ={}
            for i,face_key in enumerate(out):
                cmp_face = out[face_key]
                if i==0:
                    cv2.imwrite(equator_save_path + f"/{r}/{j:04d}.png",cmp_face)
                elif i==1:
                    cv2.imwrite(equator_save_path + f"/{r+length}/{j:04d}.png",cmp_face)
                elif i==2:
                    cv2.imwrite(equator_save_path + f"/{r+2*length}/{j:04d}.png",cmp_face)
                elif i==3:
                    cv2.imwrite(equator_save_path + f"/{r+3*length}/{j:04d}.png",cmp_face)
                elif i==4:
                    cv2.imwrite(poles_save_path + f"/{r}/{j:04d}.png",cmp_face)
                else:
                    cv2.imwrite(poles_save_path+ f"/{r+length}/{j:04d}.png", cmp_face)
    length = len(validation_videos)
    for r,video in enumerate(validation_videos):
        path1 = path_to_erp_videos_frames +"/"+video
        list = os.listdir(path1)
        for j,item in enumerate(list):
            img = os.path.join(path1,item)
            img = cv2.imread(img)
            img = np.array(img)
            out = e2c(img, face_w=160 , mode='bilinear', cube_format='dict')
            out_predict ={}
            for i,face_key in enumerate(out):
                cmp_face = out[face_key]
                if i==0:
                    print(equator_save_path + f"/{r}/{j:04d}.png")
                    cv2.imwrite( equator_save_validation_path + f"/{r}/{j:04d}.png",cmp_face)
                elif i==1:
                    cv2.imwrite( equator_save_validation_path + f"/{r+length}/{j:04d}.png",cmp_face)
                elif i==2:
                    cv2.imwrite( equator_save_validation_path + f"/{r+2*length}/{j:04d}.png",cmp_face)
                elif i==3:
                    cv2.imwrite( equator_save_validation_path + f"/{r+3*length}/{j:04d}.png",cmp_face)
                elif i==4:
                    cv2.imwrite( poles_save_validation_path+ f"/{r}/{j:04d}.png",cmp_face)
                else:
                    cv2.imwrite( poles_save_validation_path+ f"/{r+66}/{j:04d}.png", cmp_face)


