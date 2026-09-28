"""Resize pixels and XYXY boxes together; torchvision handles mean/std later."""

import torch
from torch.nn import functional as F


def preprocess(image, boxes, image_size=512):
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    image = torch.as_tensor(image, dtype=torch.float32)
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).clone()
    if image.ndim != 2 or not torch.isfinite(image).all():
        raise ValueError("Expected finite H x W pixels")
    height, width = image.shape
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixels must be normalized to [0, 1]")
    if not torch.isfinite(boxes).all():
        raise ValueError("Nonfinite bounding boxes")
    if (boxes[:, :2] < 0).any() or (boxes[:, 2:] > torch.tensor([width, height])).any():
        raise ValueError("Annotation box lies outside DICOM dimensions")
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("Box must have positive width and height")
    # Square resize uses separate x/y scales, without cropping or dropping boxes.
    boxes *= torch.tensor([image_size / width, image_size / height] * 2)
    image = F.interpolate(image[None, None], size=(image_size, image_size),
                          mode="bilinear", align_corners=False, antialias=True)[0]
    return image.repeat(3, 1, 1), boxes
