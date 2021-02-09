import os
import cv2
import numpy as np
import functools
import math


def update_idx(idx, shape):
    res_idx = [idx[0], idx[1]]
    if (idx[1] + 1) % shape[1] > 0:
        res_idx[1] += 1
    else:
        if (idx[0] + 1) % shape[0] > 0:
            res_idx[1] = 0
            res_idx[0] += 1

    return (res_idx[0], res_idx[1])

def frames_to_multiscene(list_of_frames,
                        texts=None,
                        resulting_frame_shape=(768, 1360),
                        method="grid",
                        grid_dim=None,
                        resizer_func=None):
    """
    Places multiple frames in one frame side by side

    Parameters
    ----------
    list_of _frames
        a list of frames (images) with compatible depth dimention
    texts
        a list of corresponding text titles for each frame to be 
        placed on top of it in the resulting frame
    resulting_frame_shape
        a tuple of size 2 representing width and height of resulting
        image
        Note: the output frame shape might differ slightly from this
              value
    method
        method can be one of 
            grid
            horizontal
            vertical
    resizer_func
        use 'functools.partial' to create the function descriptor
    """
    if method not in ["grid", "horizontal", "vertical"]:
        raise ValueError(f"wrong value for parameter method")

    if method == 'vertical':
        img_size = (math.floor(resulting_frame_shape[0] / len(list_of_frames)),
                    resulting_frame_shape[1],
                    3)

        resulting_frame_shape = (len(list_of_frames) * img_size[0],
                                 img_size[1],
                                 3)

        resulting_frame = np.zeros(resulting_frame_shape, np.uint8)

        idxs = (0, 0)
        for i, img in enumerate(list_of_frames):
            new_img = resizer_func(img, img_size[:-1])

            if texts:
                cv2.putText(new_img, #numpy array on which text is written
                            texts[i], #text
                            (6, 20), #position at which writing has to start
                            cv2.FONT_HERSHEY_SIMPLEX, #font family
                            1, #font size
                            (10, 10, 255, 255), #font color
                            2) # font strok

            resulting_frame[idxs[0] * img_size[0]:(idxs[0] + 1) * img_size[0],
                            idxs[1] * img_size[1]:(idxs[1] + 1) * img_size[1],
                            :] = new_img

            idxs = (idxs[0]+1, idxs[1])

    if method == 'horizontal':
        img_size = (resulting_frame_shape[0],
                    math.floor(resulting_frame_shape[1] / len(list_of_frames)),
                    3)

        resulting_frame_shape = (img_size[0],
                                 len(list_of_frames) * img_size[1],
                                 3)

        resulting_frame = np.zeros(resulting_frame_shape, np.uint8)

        idxs = (0, 0)
        for i, img in enumerate(list_of_frames):
            new_img = resizer_func(img, img_size[:-1])

            if texts:
                cv2.putText(new_img, #numpy array on which text is written
                            texts[i], #text
                            (6, 20), #position at which writing has to start
                            cv2.FONT_HERSHEY_SIMPLEX, #font family
                            1, #font size
                            (10, 10, 255, 255), #font color
                            2) # font strok

            resulting_frame[idxs[0] * img_size[0]:(idxs[0] + 1) * img_size[0],
                            idxs[1] * img_size[1]:(idxs[1] + 1) * img_size[1],
                            :] = new_img

            idxs = (idxs[0], idxs[1]+1)

    if method == 'grid':
        if not grid_dim:
            grid_dim = math.floor(math.sqrt(len(list_of_frames)))
            if grid_dim**2 < len(list_of_frames):
                grid_dim += 1
            grid_dim = (grid_dim, grid_dim)
        
        img_size = (math.floor(resulting_frame_shape[0] / grid_dim[0]),
                    math.floor(resulting_frame_shape[1] / grid_dim[1]),
                    3)

        resulting_frame_shape = (grid_dim[0] * img_size[0],
                                 grid_dim[1] * img_size[1],
                                 3)

        resulting_frame = np.zeros(resulting_frame_shape, np.uint8)

        idxs = (0, 0)
        for i, img in enumerate(list_of_frames):
            new_img = resizer_func(img, img_size[:-1])
            if texts:
                cv2.putText(new_img, #numpy array on which text is written
                            texts[i], #text
                            (6, 20), #position at which writing has to start
                            cv2.FONT_HERSHEY_SIMPLEX, #font family
                            .5, #font size
                            (10, 10, 255, 255), #font color
                            1) # font strok

            resulting_frame[idxs[0] * img_size[0]:(idxs[0] + 1) * img_size[0],
                            idxs[1] * img_size[1]:(idxs[1] + 1) * img_size[1],
                            :] = new_img

            idxs = update_idx(idxs, grid_dim)

    return resulting_frame, resulting_frame_shape

def get_cv_resize_function():
    def wraper(img, shape):
        new_img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        return new_img
    
    return wraper

def bbox_from_mask(mask, image=None, threshold=None):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if image is not None:
        result = image.copy()
        if len(result.shape) < 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            if threshold:
                if w < threshold and h < threshold:
                    continue
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return result
    else:
        bboxes = [cv2.boundingRect(cntr) for cntr in contours]
        return bboxes

def rescal_to_image(frame, subtract_min=False):
    x = np.array(frame, np.float32)

    if subtract_min:
        x = x - (np.min(x) if np.min(x) < 0 else 0)
    x = x / np.max(x)
    x = x * 255
    x = np.floor(x)

    x = x.astype('uint8')
    return x
