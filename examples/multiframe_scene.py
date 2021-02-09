import os
import cv2
import numpy as np
import functools
import math

import sys
sys.path.insert(0, "/home/devuser/Documents/workspace/camera/pylon_camera_tools/")

from PylonCameraTools.IO import HuffyuvLosslessReader, HuffyuvLosslessSaver, \
    H264LossLessReader, H264LossLessSaver

sys.path.insert(0, "/home/devuser/Documents/workspace/video_toolkit/")

from VideoToolkit import frames_to_multiscene, get_cv_resize_function, \
                    rescal_to_image



if __name__ == "__main__":
    filename_list = ["/home/devuser/Documents/workspace/data/experiment3_400-400.m4v",
                    "/home/devuser/Documents/workspace/background_subtraction/opencv_bs/CNT.m4v",
                    "/home/devuser/Documents/workspace/background_subtraction/opencv_bs/MOG2.m4v",
                    "/home/devuser/Documents/workspace/background_subtraction/opencv_bs/KNN.m4v",
                    "/home/devuser/Documents/workspace/background_subtraction/Adaptive-foreground-background-segmentation-using-GMMs/AFBSuGMM.m4v",
                    "/home/devuser/Documents/workspace/background_subtraction/ViBe/vibe_python/vibe1.m4v"
                    ]
    texts = ["main",
            "CNT",
            "MOG2",
            "KNN",
            "AFBSuGMM",
            "ViBe"
            ]
    reader_list = [H264LossLessReader(input_filename=filename,
                                width_height=None,
                                fps=None) for filename in filename_list]

    writer = None

    resizer_func = get_cv_resize_function()
    res_frame_shape = (600, 1024) # width, height

    flag = True
    count = 0
    while True:
        frame_list = []
        for reader in reader_list:
            frame = reader.read()
            if frame is None:
                print("*******Got None frame")
                flag = False
            frame_list.append(frame)
        if not flag:
            break

        print("*********************", frame_list[0].shape)
        
        res_frame, res_shape = frames_to_multiscene(frame_list,
                                                    texts=texts,
                                                    resulting_frame_shape=res_frame_shape,
                                                    method='grid',
                                                    grid_dim=(2, 3),
                                                    resizer_func=resizer_func)
    
        if writer is None:
            writer = H264LossLessSaver("multiframe",
                                        (res_shape[1], res_shape[0]),
                                        20,
                                        compression_rate=25)

        print("res_frame ", res_frame.shape)
        writer.write(res_frame)

        cv2.imshow('image', res_frame)
        
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            breakeyboard = cv2.waav
            
        # if count > 100:
        #     break
        # count +=1

    
    for i, _ in enumerate(reader_list):
        del reader_list[i]
    del writer
