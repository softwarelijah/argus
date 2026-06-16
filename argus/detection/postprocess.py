"""Pure-numpy pre/post processing for raw YOLOv8 engine output.

These helpers are needed by the TensorRT path, where there is no Ultralytics
wrapper to do letterboxing and NMS for us.
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (1280, 1280),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize and pad an image to ``new_shape`` while keeping aspect ratio.

    Returns the padded image, the scale ratio and the (left, top) padding so
    boxes can be mapped back to the original frame.
    """
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def preprocess(
    image: np.ndarray, imgsz: int = 1280
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox, BGR->RGB, HWC->CHW, normalise to [0, 1] and add a batch dim."""
    padded, ratio, pad = letterbox(image, (imgsz, imgsz))
    blob = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
    return blob[None], ratio, pad


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy non-maximum suppression. Boxes are (N, 4) xyxy."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(
    output: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.7,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YOLOv8 output into boxes, scores and class ids.

    YOLOv8 exports a ``(1, 4 + num_classes, num_anchors)`` tensor. We transpose
    it, threshold on the best class score, run NMS and unmap the boxes back to
    the original image coordinates.
    """
    pred = np.squeeze(output, axis=0).T  # -> (num_anchors, 4 + num_classes)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4 : 4 + num_classes]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    keep_mask = confidences > conf_thresh
    boxes_xywh = boxes_xywh[keep_mask]
    confidences = confidences[keep_mask]
    class_ids = class_ids[keep_mask]
    if boxes_xywh.shape[0] == 0:
        return (np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))

    # cx, cy, w, h -> x1, y1, x2, y2
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Undo letterbox padding and scaling.
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= ratio

    keep = nms(boxes, confidences, iou_thresh)
    return boxes[keep].astype(np.float32), confidences[keep].astype(np.float32), class_ids[
        keep
    ].astype(np.int32)
