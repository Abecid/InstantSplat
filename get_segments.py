import os
import json
from os import makedirs
from time import time, perf_counter
from argparse import ArgumentParser
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "models"))

import torch
import torchvision
from tqdm import tqdm
import imageio
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import cv2
from rembg import remove
from PIL import Image

# VCO
from config import make_vco_args, init_cam_configs
from modules.segmentation import SegmentationMap
from modules.objectDetect import ObjectDetector
from interface.common import CameraView
from modules.outputData import OutputData

imgs_path = "assets/welstory/362/images"
output_path = "output/welstory/362/segmentations"
bbox_output_path = "output/welstory/362/bboxes"
overlay_segments_output_path = "output/welstory/362/overlay_segments"
rmbeg_output_path = "output/welstory/362/rembg"

dataset_type="welstory_1st"

mannamil_dataset_path = "dataset/0000_mannamil_20250707_061547/raw"
vco_2nd_dataset_path = "dataset/vco_2nd_eval/data/vco_7/log/capture"
welstory_dataset_path = "dataset/vco_welstory_pre_eval/data/vco_vol_3/log/capture"
welstory_poc_dataset_path = "dataset/vco_eval_welstory_1st_poc/data/vco_vol_4/log/capture"

if dataset_type == "mannamil":
    data_id = "937_135"
    dataset_path = f"{mannamil_dataset_path}/{data_id}"
elif dataset_type == "vco_2nd":
    data_id = "0"
    dataset_path = f"{vco_2nd_dataset_path}/{data_id}"
elif dataset_type == "welstory_1st":
    data_id = "9"
    dataset_path = f"{welstory_poc_dataset_path}/{data_id}"

def vco_setup():
    vco_args = make_vco_args(overrides={
        "main_cam": "TB",
        "use_detection_segmentation": "true",
        "prompt_type": "box",
        "sam2_type": "imagepred",
        "image_height": 480,
        "image_width": 640,
        "object_detector": "yolo",
        "store_cd": "welstory_1st",
        "num_top_k": 2,
        "depth_threshold": 30.0,
    })
    if dataset_type == "mannamil":
        vco_args.stereo_path = f"{dataset_path}/mask/stereo_config_online.json"
        mask_path = f"{dataset_path}/mask"
        vco_args.object_detector = "dfine"
        vco_args.project = "manna"
    elif dataset_type == "vco_2nd":
        mask_path = dataset_path[:dataset_path.find("/log")] + "/mask"
        vco_args.stereo_path = dataset_path[:dataset_path.find("/data/vco")] + "/stereo_config_online.json"
        vco_args.object_detector = "yolo"
        vco_args.project = "phase3"
    elif dataset_type == "welstory_1st":
        vco_args.stereo_path = dataset_path[:dataset_path.find("/log")] + "/mask/stereo_config_online.json"
        mask_path = dataset_path[:dataset_path.find("/log")] + "/mask"
        vco_args.object_detector = "yolo"
        vco_args.store_cd = "welstory_1st"

    valid_camera_views: list[CameraView] = [
        CameraView.TOP_BACK,
        CameraView.TOP_FRONT,
        CameraView.TOP_LEFT,
        CameraView.TOP_RIGHT,
        CameraView.TOP_CENTER,
        CameraView.LEFT_WING,
        CameraView.RIGHT_WING,
    ]

    roi_mask_coords = init_cam_configs(
        valid_camera_views, cam_coords_root=mask_path
    )

    return vco_args, roi_mask_coords

def load_images(imgs_path):
    images = {}
    for fname in os.listdir(imgs_path):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(imgs_path, fname)
            image = np.array(cv2.imread(image_path))
            images[fname] = image  # keep filename for sorting
    return images

def main():
    config, roi_mask_coords = vco_setup()
    
    valid_vco_camera_views = [
        'LW',
        'TC',
        'RW',
    ]
    output_data = OutputData(config)
    yolo_masks = dict()
    output_data.cls = defaultdict(list)
    output_data.conf = defaultdict(list)
    output_data.visualize_object_detector = defaultdict(list)
    bg_id = (0, 42, 43, 44, 45, 46, 47)
    is_remove_bg = True

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(overlay_segments_output_path, exist_ok=True)
    os.makedirs(bbox_output_path, exist_ok=True)
    os.makedirs(rmbeg_output_path, exist_ok=True)

    # Read image by iterating over the input_path directory
    images = load_images(imgs_path)

    object_detector = ObjectDetector(
        config, cam_coords=roi_mask_coords
    )

    for i, img_name in enumerate(images):
        occupancy_check = False
        image = images[img_name]
        vco_cam_view = img_name.split("_")[2]

        (
            output_data.objectbox[vco_cam_view],
            output_data.numobject[vco_cam_view],
            output_data.occupancy,
            yolo_masks[vco_cam_view],
        ) = object_detector(
            image,
            # output_data,
            vco_cam_view,
            mode="capture",
            occupancy_check=occupancy_check,
            bg_id=bg_id,
            is_remove_bg=is_remove_bg,
        )

        bboxes = output_data.objectbox[vco_cam_view]
        bbox_image = image.copy()
        # Draw bounding boxes on the image
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Save the image with bounding boxes
        output_image_path = os.path.join(bbox_output_path, img_name.split(".")[0] + ".png")
        cv2.imwrite(output_image_path, bbox_image)
        print(f"Saved image with bounding boxes: {output_image_path}")

    # Run semantic segmentation
    segmentationmap = SegmentationMap(
        window=None,
        config=config,
        main_cam_view=config.main_cam,
        cam_coords=roi_mask_coords,
    )

    for i, img_name in enumerate(images):
        image = images[img_name]
        vco_cam_view = img_name.split("_")[2]

        output_data = segmentationmap(
            image,
            output_data,
            vco_cam_view,
            mode="capture",
            occupancy_check=False,
            yolo_masks=yolo_masks[vco_cam_view],
            use_detection_segmentation=config.use_detection_segmentation,
        )

        segmentmap = output_data.multisegmentlist[vco_cam_view][0][0]
        # Draw segmentation map on the image
        colored_mask = (segmentmap * 255).astype(np.uint8)
        colored_mask = cv2.applyColorMap(colored_mask, cv2.COLORMAP_JET)
        overlay_image = image.copy()
        overlay = cv2.addWeighted(overlay_image, 0.7, colored_mask, 0.3, 0)
        # Save the image with segmentation map
        output_image_path = os.path.join(overlay_segments_output_path, img_name.split(".")[0] + ".png")
        cv2.imwrite(output_image_path, overlay)
        print(f"Saved image with segmentation map: {output_image_path}")

        # Save rgba segmentation image
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        background_mask = segmentmap == 0
        output_image[background_mask, 3] = 0 # Index 3 is the alpha channel

        # 5. Save the final image with a transparent background
        # The file MUST be saved as a .png to preserve transparency
        output_image_path = os.path.join(output_path, img_name.split(".")[0] + ".png")
        cv2.imwrite(output_image_path, output_image)

        # Run rembg to remove background    
        input_pil = Image.open(output_image_path)
        output_pil = remove(input_pil)
        output_image_path = os.path.join(rmbeg_output_path, img_name.split(".")[0] + ".png")
        output_pil.save(output_image_path)
        print(f"Saved rgba segmentation image: {output_image_path}")

if __name__ == "__main__":
    main()