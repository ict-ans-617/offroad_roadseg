import cv2  # type: ignore

# from segment_anything import sam_model_registry

import numpy as np
import torch
import glob
import time
import os
import matplotlib
import matplotlib.pyplot as plt
# from typing import Any, Dict, List
from typing import Any, Dict, List, Tuple

import zipfile


from seg_decoder import SegHead, SegHeadUpConv
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F


from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

from demo import show_anns, show_anns_2, preprocess, postprocess_masks

from common import visualize

from video_utils import open_video_capture, read_video_frame

def main():
    print("Loading model...")
    # sam_model = sam_model_registry['vit_h'](checkpoint='/raid/yehongliang_data/SAM_ckpts/sam_vit_h_4b8939.pth')
    # sam_model = sam_model_registry['vit_l'](checkpoint='/raid/yehongliang_data/SAM_ckpts/sam_vit_l_0b3195.pth')
    # sam_model = sam_model_registry['vit_h'](checkpoint='/home/zj/code/kyxz_sam_roadseg/SAM-checkpoint/sam_vit_h_4b8939.pth')

    # efficientsam = build_efficient_sam_vitt()
    efficientsam = build_efficient_sam_vits()

    # device = torch.device("cuda:2")
    device = torch.device("cuda:0")

    efficientsam.to(device)

    seg_decoder = SegHead()
    ckpt_path = 'ckpts/orfd/best_epoch.pth'
    seg_decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    seg_decoder.eval()
    seg_decoder.to(device)

    video_capture = open_video_capture()


    transform = ResizeLongestSide(1024)


    while True:
        ret, frame = read_video_frame(video_capture)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = frame
        # print("Image shape:", image.shape)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = rgb_image.shape[:2]

        start_time = time.time()
        print(f"{start_time = }")
        input_image = transform.apply_image(rgb_image)
        input_size = input_image.shape[:2]

        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # input_image_torch = efficientsam.preprocess(input_image_torch)
        input_image_torch = preprocess(input_image_torch)
        input_image_torch = input_image_torch.to(device)
        


        with torch.no_grad():
            start_encode_time = time.time()
            image_embedding = efficientsam.image_encoder(input_image_torch)
            middle_time = time.time()
            encode_time = middle_time - start_encode_time
            pred_mask = seg_decoder(image_embedding)
            decode_time = time.time() - middle_time
            pred_mask = postprocess_masks(pred_mask, input_size, ori_size)

        print(f"{encode_time = }")
        print(f"{decode_time = }")
        infer_time = time.time() - start_time
        print(f"{infer_time = }")
        # fps = 1 / infer_time
        # print(f"{fps = }")

        image = visualize(pred_mask=pred_mask, image=image)

        cv2.imshow("frame", image)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()