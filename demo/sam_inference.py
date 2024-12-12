import gc

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import gradio as gr
import cv2
from demo.mask_utils import *

class SAM_Inference:
    def __init__(self, model_type='vit_b', device='cuda') -> None:
        models = {
            'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
            'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
            'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
        }

        sam = sam_model_registry[model_type](checkpoint=models[model_type])
        sam = sam.to(device)

        self.predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(model=sam)

    def img_select_point(self, original_img: np.ndarray, evt: gr.SelectData):
        img = original_img.copy()
        sel_pix = [(evt.index, 1)]  # append the foreground_point

        masks = self.run_inference(original_img, sel_pix)
        for point, label in sel_pix:
            cv2.circle(img, point, 5, (240, 240, 240), -1, 0)
            cv2.circle(img, point, 5, (30, 144, 255), 2, 0)

        mask = masks[0][0]
        colored_mask = mask_foreground(mask)
        res = img_add_masks(original_img, colored_mask, mask)
        return img, process_mask_to_show(mask), res, mask

    def gen_box_seg(self, inp):
        if inp is None:
            raise gr.Error("Please upload an image first!")
        image = inp['image']
        if len(inp['boxes']) == 0:
            raise gr.Error("Please clear the raw boxes and draw a box first!")
        boxes = inp['boxes'][-1]

        input_box = np.array([boxes[0], boxes[1], boxes[2], boxes[3]]).astype(int)

        masks = self.predict_box(image, input_box)

        mask = masks[0][0]
        colored_mask = mask_foreground(mask)
        res = img_add_masks(image, colored_mask, mask)

        return process_mask_to_show(mask), res, mask
    
    def gen_box_point(self, inp):
        if inp is None:
            raise gr.Error("Please upload an image first!")
        image = inp['image']
        if len(inp['boxes']) == 0:
            raise gr.Error("Please clear the raw boxes and draw a box first!")
        boxes = inp['boxes'][-1]

        input_box = np.array([boxes[0], boxes[1], boxes[2], boxes[3]]).astype(int)
        mask = np.zeros_like(image[:,:,0])

        mask[input_box[1]:input_box[3], input_box[0]:input_box[2]] = 1  # generate the mask based on bbox points

        colored_mask = mask_foreground(mask)
        res = img_add_masks(image, colored_mask, mask)

        return process_mask_to_show(mask), res, mask

    
    def run_inference(self, input_x, selected_points):
        if len(selected_points) == 0:
            return []

        self.predictor.set_image(input_x)

        points = torch.Tensor(
            [p for p, _ in selected_points]
        ).to(self.predictor.device).unsqueeze(0)

        labels = torch.Tensor(
            [int(l) for _, l in selected_points]
        ).to(self.predictor.device).unsqueeze(0)

        transformed_points = self.predictor.transform.apply_coords_torch(
            points, input_x.shape[:2])

        # predict segmentation according to the boxes
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=transformed_points,
            point_labels=labels,
            multimask_output=False,
        )
        masks = masks.cpu().detach().numpy()

        gc.collect()
        torch.cuda.empty_cache()

        return masks

    def predict_box(self, input_x, input_box):
        self.predictor.set_image(input_x)

        input_boxes = torch.tensor(input_box[None, :], device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, input_x.shape[:2])

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        masks = masks.cpu().detach().numpy()

        gc.collect()
        torch.cuda.empty_cache()
        return masks
