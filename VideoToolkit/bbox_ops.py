import numpy as np


# Implementation is originally from : https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
# with some modifications.
def _box_cxcywh_to_xyxy(box: tuple) -> tuple:
    """Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box

    Parameters
    ----------
    box : tuple
        box in (cx, cy, w, h) format which will be converted.

    Returns
    -------
    tuple
        box in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = box
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return (x1, y1, x2, y2)

def _box_xyxy_to_cxcywh(box: tuple) -> tuple:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box

    Parameters
    ----------
    box : tuple
        boxes in (x1, y1, x2, y2) format which will be converted.

    Returns
    -------
    tuple
        boxes in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return (cx, cy, w, h)

def _box_xywh_to_xyxy(box: tuple) -> tuple:
    """Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
    (x, y) refers to top left of bouding box.
    (w, h) refers to width and height of box.

    Parameters
    ----------
    box : tuple
        box in (x, y, w, h) which will be converted.

    Returns
    -------
    tuple
        box in (x1, y1, x2, y2) format.
    """
    x, y, w, h = box
    return (x, y, x + w, y + h)

def _box_xyxy_to_xywh(box: tuple) -> tuple:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box

    Parameters
    ----------
    box : tuple
        box in (x1, y1, x2, y2) which will be converted.

    Returns
    -------
    tuple
        box in (x, y, w, h) format.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1  # x2 - x1
    h = y2 - y1  # y2 - y1
    
    return (x1, y1, w, h)

def convert_bbox(box: tuple, in_fmt: str, out_fmt: str) -> tuple:
    """Converts boxes from given in_fmt to out_fmt.

    Parameters
    ----------
    boxe : tuple
        Box which will be converted.
    in_fmt : str
        Input format of given box. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        Supported in_fmt and out_fmt are:
            'xyxy': box are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
            'xywh' : box are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
            'cxcywh' : box are represented via centre, width and height, cx, cy being center of box, w, h
            being width and height.
    out_fmt : str
        Output format of given box. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns
    -------
    tuple
        Box into converted format.

    Raises
    ------
    ValueError
        [description]
    """

    allowed_fmts = ("xyxy", "xywh", "cxcywh")
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError("Unsupported Bounding Box Conversions for given in_fmt and out_fmt")

    if in_fmt == out_fmt:
        return box

    # convert to xyxy
    if in_fmt == "xywh":
        box = _box_xywh_to_xyxy(box)
    elif in_fmt == "cxcywh":
        box = _box_cxcywh_to_xyxy(box)
    
    if out_fmt == 'xyxy':
        return box
    elif out_fmt == "xywh":
        box = _box_xyxy_to_xywh(box)
    elif out_fmt == "cxcywh":
        box = _box_xyxy_to_cxcywh(box)
    return box
####################################################

# Code from : https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py
# with slight modifications
def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    px, py, pxp, pyp = pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]
    gx, gy, gxp, gyp = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
    # 1.get the coordinate of inters
    ixmin = max(px, gx)
    ixmax = min(pxp, gxp)
    iymin = max(py, gy)
    iymax = min(pyp, gyp)

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = ((pxp - px + 1.) * (pyp - py + 1.) +
           (gxp - gx + 1.) * (gyp - gy + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

# ToDo: fix needed
def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax
####################################################

def get_overlapping(pred_box, gt_box):
    """Computes ratio of overlapping area of two bounding boxes
    to each of them and returns the maximum value (in percentage)

    Parameters
    ----------
    pred_box : Tuple
        A tuple of (x, y, w, h) where 'x' is column index and 'y' is row index
    gt_box : Tuple
        Same formatting as 'pred_box'

    Returns
    -------
    float
        maximium ratio of overlapping area to both bboxes
    """
    px, py, pxp, pyp = pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]
    pw, ph = pred_box[2], pred_box[3]
    gx, gy, gxp, gyp = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
    gw, gh = gt_box[2], gt_box[3]
    # 1.get the coordinate of inters
    ixmin = max(px, gx)
    ixmax = min(pxp, gxp)
    iymin = max(py, gy)
    iymax = min(pyp, gyp)

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of each bboxs
    parea = pw * ph
    garea = gw * gh

    return max(inters / parea, inters / garea)

def merge_bboxs(pred_box, gt_box):
    """merges two bounding boxes into one circumscribing box

    Parameters
    ----------
    pred_box : Tuple
        A tuple of (x, y, w, h) where 'x' is column index and 'y' is row index
    gt_box : Tuple
        A tuple of (x, y, w, h) where 'x' is column index and 'y' is row index

    Returns
    -------
    Tuple
        A tuple of (x, y, w, h) where 'x' is column index and 'y' is row index
    """    
    px, py, pxp, pyp = pred_box[0], pred_box[1], pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]
    pw, ph = pred_box[2], pred_box[3]
    gx, gy, gxp, gyp = gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
    gw, gh = gt_box[2], gt_box[3]
    # 1.get the coordinate of inters
    ixmin = min(px, gx)
    ixmax = max(pxp, gxp)
    iymin = min(py, gy)
    iymax = max(pyp, gyp)

    return ixmin, iymin, ixmax - ixmin, iymax - iymin

def bbox_area(bbox):
    """calculates area of a bounding box

    Parameters
    ----------
    bbox : Tuple
        A tuple of (x, y, w, h) where 'x' is column index and 'y' is row index  

    Returns
    -------
    Integer
        the area of given bounding box
    """
    return bbox[2] * bbox[3]

def bbox_overlapping_sets(bboxes, threshold=0.5):
    # ToDo: get arguments for iot and overlap
    areas = map(bbox_area, bboxes)
    pool = [bboxes[i] for i in np.argsort(areas)]

    res = []
    while len(pool) > 0:
        res.append([pool[0]])
        del pool[0]

        ref = res[-1][0]
        del_idx = []
        for i, box in enumerate(pool):
            overlap = get_overlapping(ref, box)
            iou = get_iou(ref, box)
            # print(f"overlavp {overlap}, IOU {iou} ")
            if overlap > threshold or iou > threshold:
                res[-1].append(box)
                del_idx.append(i)

        for i in del_idx[::-1]:
            del pool[i]

    return res

def arg_bbox_overlapping_sets(bboxes, threshold=0.5):
    # ToDo: get arguments for iot and overlap
    areas = [bbox_area(bbox) for bbox in bboxes]
    pool = np.argsort(areas).tolist()

    res = []
    while len(pool) > 0:
        res.append([pool[0]])
        del pool[0]

        ref = res[-1][0]
        del_idx = []
        for i, box in enumerate(pool):
            overlap = get_overlapping(bboxes[ref], bboxes[box])
            iou = get_iou(bboxes[ref], bboxes[box])
            # print(f"overlavp {overlap}, IOU {iou} ")
            if overlap > threshold or iou > threshold:
                res[-1].append(box)
                del_idx.append(i)
        for i in del_idx[::-1]:
            del pool[i]

    return res

def bbox_non_maxima_union(bboxes, threshold=0.5):
    """"""
    # ToDo: get arguments for iot and overlap
    oset = bbox_overlapping_sets(bboxes, threshold)

    res = []
    for box_set in oset:
        ref = box_set[0]
        for box in box_set[1:]:
            ref = merge_bboxs(ref, box)
        res.append(ref)

    return res

def fix_dim(x, w, new_w, original_w):
    new_x = x - ((new_w - w) / 2)
    if new_x + new_w >= original_w:
        new_x -= (new_x + new_w) - original_w
    if new_x < 0:
        new_x = 0

    return new_x


def expand_bbox(bbox: tuple, shape: tuple, to_size=None, padding=None) -> tuple:
    """[summary]

    Parameters
    ----------
    bbox : tuple
        (x, y, x1, y1) of the bounding box
    shape : tuple
        (width, height) of the original image
    to_size : int or tuple, optional
        (width, height) of the resulting image, by default None
    padding : int, optional
        ammount of padding, by default None

    Returns
    -------
    tuple
        (x, y, x1, y1) of the resulting bounding box
    """
    assert to_size is None or padding is None, \
            "Both of the parameters 'to_size' and 'padding' are passed." + \
            " only one of them is allowed."
    assert not (to_size is not None and padding is not None), \
            "One of the parameters 'to_size' or 'padding' is required"

    x, y, w, h = convert_bbox(bbox, in_fmt='xyxy', out_fmt='xywh')

    ow = shape[0]
    oh = shape[1]

    if to_size is not None:
        if isinstance(to_size, tuple) or isinstance(to_size, list):
            rw = to_size[0]
            rh = to_size[1]
        else:
            rw = rh = to_size
        assert ow >= rw and oh >= rh, \
            "Inner requested box is bigger than the original image size."
        
        # fix width
        rx = fix_dim(x, w, rw, ow)
        # fix height
        ry = fix_dim(y, h, rh, oh)

        res_box = (int(rx), int(ry), int(rw), int(rh))
        return convert_bbox(res_box, in_fmt='xywh', out_fmt='xyxy')

    if padding is not None:
        rx = max(x - padding, 0)
        ry = max(y - padding, 0)
        rw = min(w + rx + 2*padding, ow) - rx
        rh = min(h + ry + 2*padding, oh) - ry

        res_box = (int(rx), int(ry), int(rw), int(rh))
        return convert_bbox(res_box, in_fmt='xywh', out_fmt='xyxy')

def crop_patch(img: np.ndarray, bbox: tuple) -> np.ndarray:
    """crops a patch from the image given bounding box of the pathc

    Parameters
    ----------
    img : np.ndarray
        input image
    bbox : tuple
        bounding box in (x, y, x1, y1) format

    Returns
    -------
    np.ndarray
        cropped patch from image
    """    
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
