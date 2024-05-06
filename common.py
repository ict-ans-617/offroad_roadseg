
import torch
from torch.nn import functional as F
import time
import numpy as np
import cv2


def visualize(pred_mask, image):
    start_visual_time = time.time()
    # print(f"{start_visual_time = }")
    downsample_rate = 1.0
    pred_mask = F.interpolate(pred_mask.detach(), scale_factor=downsample_rate)
    pred = torch.softmax(pred_mask, dim=1).float()[0, 1]
    pred = pred > 0.3

    start_to_cpu_time = time.time()
    pred = pred.detach().cpu().numpy()
    to_cpu_time = time.time() - start_to_cpu_time
    print(f"{to_cpu_time = }")
    pred = np.repeat(pred[..., None], 3, axis=-1)
    pred_color = pred * np.array([255, 0, 85], dtype=np.uint8)

    image = cv2.resize(image, dsize=(0, 0), fx=downsample_rate, fy=downsample_rate)
    image = np.where(pred, pred_color, image)

    visual_time = time.time() - start_visual_time
    print(f"{visual_time = }")
    return image
