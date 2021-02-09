import glob
import os
import cv2
import numpy as np
import sys
sys.path.insert(0, "/home/devuser/Documents/workspace/camera/pylon_camera_tools/")

from PylonCameraTools.IO import HuffyuvLosslessReader, HuffyuvLosslessSaver, \
    H264LossLessReader, H264LossLessSaver



def rescal_to_image(frame, subtract_min=False):
    x = np.array(frame, np.float32)

    if subtract_min:
        x = x - (np.min(x) if np.min(x) < 0 else 0)
    x = x / np.max(x)
    x = x * 255
    x = np.floor(x)

    x = x.astype('uint8')
    return x

def get_idx(name):
    name_ls = os.path.splitext(os.path.split(name)[1])[0].split('_')
    name_idx = name_ls[-1]
    return int(name_idx)

def get_file_list(base_dir):
    dir_list = glob.glob(base_dir + "/*.tiff")
    name_ls = os.path.splitext(os.path.split(dir_list[0])[1])[0].split('_')
    name = '_'.join(name_ls[:-1])
    idx_list = [int(get_idx(d)) for d in dir_list]
    idx_list.sort()
    dir_list = [name + f"_{i}.tiff" for i in idx_list]
    return dir_list


if __name__ == "__main__":
    fps = 25
    base_dir = "/home/devuser/Documents/workspace/data/3000-01"

    files_list = get_file_list(base_dir)
    
    img = cv2.imread(os.path.join(base_dir, files_list[0]))
    img = np.swapaxes(img, 1, 0)

    writer = H264LossLessSaver(os.path.join(base_dir, "video"),
                                img.shape,
                                fps)

    for f in files_list:
        if os.path.exists(os.path.join(base_dir, f)):
            img = cv2.imread(os.path.join(base_dir, f))
        else:
            print("path is wrong")
        # img = np.swapaxes(img, 1, 0)
        img = rescal_to_image(img)
        writer.write(img)

        # print(img.dtype, img.shape)
        # cv2.imshow('image', img)
        
        # keyboard = cv2.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     breakeyboard = cv2.waav

    del writer
