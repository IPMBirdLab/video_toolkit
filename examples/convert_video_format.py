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
    filename = "/home/devuser/Documents/workspace/data/5000-02/video.m4v"
    reader = H264LossLessReader(input_filename=filename,
                                width_height=None,
                                fps=None)

    writer = H264LossLessSaver("/home/devuser/Documents/workspace/data/5000-02/converted",
                                (reader.width, reader.height),
                                reader.fps,
                                compression_rate=15)

    flag = True
    count = 0
    while True:
        frame = reader.read()
        if frame is None:
            print("*******Got None frame")
            break

        writer.write(frame)

        cv2.imshow('image', frame)
        
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            breakeyboard = cv2.waav
            
        # if count > 100:
        #     break
        # count +=1

    
    del reader
    del writer
