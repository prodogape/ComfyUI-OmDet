import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import folder_paths
import json
import cv2

import logging
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from urllib.parse import urlparse
from torch.hub import download_url_to_file

from omdet.infernece.det_engine import DetEngine
from omdet.utils.plots import Annotator

logger = logging.getLogger("ComfyUI-OmDet")

OmDet_model_list = {
    "OmDet-Turbo_tiny_SWIN_T.pth": {
        "model_url": "https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T/resolve/main/OmDet-Turbo_tiny_SWIN_T.pth?download=true",
    },
    "ViT-B-16.pt": {
        "model_url": "https://huggingface.co/omlab/OmDet-Turbo_tiny_SWIN_T/resolve/main/ViT-B-16.pt?download=true",
    },
}

def download_model(model_url, model_file_path):
    try:
        torch.hub.download_url_to_file(model_url, model_file_path)
    except Exception as e:
        print(f"Failed to download model from {model_url}. Error: {e}")

def check_and_download_models():
    model_path = os.path.join(folder_paths.models_dir, "OmDet")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for model_name, model_info in OmDet_model_list.items():
        model_filename = model_name
        model_file_path = os.path.join(model_path, model_filename)

        if not os.path.exists(model_file_path):
            print(f"{model_filename} not found. Downloading from {model_info['model_url']}...")
            download_model(model_info["model_url"], model_file_path)

# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    # Larger video files were taking >.5 seconds to hash even when cached,
    # so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    results = tgt["results"]

    res_mask = []
    res_image = []

    box_color = (255, 0, 0)  # Red color for the box
    text_color = (255, 255, 255)  # White color for the text

    draw = ImageDraw.Draw(image_pil)

    # Get the current file path and use it to create a relative path to the font file
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_file_path, "docs", "PingFangRegular.ttf")
    font_size = 20 
    font = ImageFont.truetype(font_path, font_size)

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }

    for result in results:
        for item in result:
            x1,y1,x2,y2 = item['xmin'],item['ymin'],item['xmax'],item['ymax']
            label = item["label"]
            threshold = round(item["conf"], 2)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            points = [[x1, y1], [x2, y2]]

            # Save labelme json
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
            labelme_data["shapes"].append(shape)

            # Change label
            label = label + ":" + str(threshold)
            shape["threshold"] = str(threshold)

            # Draw rectangle on the image using PIL
            draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)

            # Draw label on the image using PIL
            text_size = draw.textsize(label, font=font)
            label_ymin = max(y1, text_size[1] + 10)
            draw.rectangle([(x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1)], fill=box_color)
            draw.text((x1, y1 - text_size[1] - 10), label, font=font, fill=text_color)

            # Draw mask
            mask = np.zeros((H, W, 1), dtype=np.uint8)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
            res_mask.append(mask_tensor)

    if len(res_mask) == 0:
        mask = np.zeros((H, W, 1), dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    # Convert the PIL image back to a numpy array
    image_with_boxes = np.array(image_pil)

    # Convert the modified image to a torch tensor
    image_with_boxes_tensor = torch.from_numpy(image_with_boxes.astype(np.float32) / 255.0)
    image_with_boxes_tensor = torch.unsqueeze(image_with_boxes_tensor, 0)
    res_image.append(image_with_boxes_tensor)

    return res_image, res_mask, labelme_data

def OmDet_detect(image_pil, labels, conf_threshold=0.30, nms_threshold=0.5,device="cuda"):
    prompt = "Detect {}.".format(",".join(labels))
    engine = DetEngine(batch_size=1, device=device)


    
    res = engine.inf_predict(
        "OmDet-Turbo_tiny_SWIN_T",
        task=prompt,
        data=[image_pil],
        labels=labels,
        src_type="pil",  # type of the image_paths, "local/tensor/pil/"url"
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )
    
    print("labels",labels)
    print("prompt",prompt)
    
    return res

def get_labels(input_string):
    modified_string = input_string.replace(", ", ",").replace("，", ",")
    return modified_string.split(",")

class ApplyOmDet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"default": "person,cat,dog"},
                ),
                "conf_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01},
                ),
                "nms_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01},
                ),
                "device": (["cuda", "cpu"],),
            },
        }

    CATEGORY = "ComfyUI-OmDet"
    FUNCTION = "main"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "JSON",
    )

    def main(
        self,
        image,
        prompt,
        conf_threshold,
        nms_threshold,
        device
    ):
        check_and_download_models()

        res_images = []
        res_masks = []
        res_labels = []

        for item in image:
            image_pil = Image.fromarray(np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")
            labels = get_labels(prompt)
            results = OmDet_detect(image_pil, labels, conf_threshold, nms_threshold, device)
            print('results',results)

            size = image_pil.size
            pred_dict = {
                "size": [size[1], size[0]],
                "results":results
            }

            image_tensor, mask_tensor, labelme_data = plot_boxes_to_image(image_pil, pred_dict)

            res_images.extend(image_tensor)
            res_masks.extend(mask_tensor)
            res_labels.append(labelme_data)

            if len(res_images) == 0:
                res_images.extend(item)
            if len(res_masks) == 0:
                mask = np.zeros((height, width, 1), dtype=np.uint8)
                empty_mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
                res_masks.extend(empty_mask)

        return (
            torch.cat(res_images, dim=0),
            torch.cat(res_masks, dim=0),
            res_labels,
        )
