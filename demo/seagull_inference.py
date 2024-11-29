import torch
from seagull.utils import disable_torch_init
from transformers import AutoTokenizer, CLIPImageProcessor
from seagull.model.language_model.seagull_llama import SeagullLlamaForCausalLM
from seagull.mm_utils import tokenizer_image_token
from seagull.conversation import conv_templates, SeparatorStyle
from seagull.constants import IMAGE_TOKEN_INDEX
from seagull.train.train import DataArguments

from functools import partial
import os
import numpy as np
import cv2
from typing import List
from PIL import Image

class Seagull():
    def __init__(self, model_path, device='cuda'):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048, padding_side="right", use_fast=True)
        self.model = SeagullLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,).to(device)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_processor = CLIPImageProcessor(do_resize=True, size={"shortest_edge":512}, resample=3,  do_center_crop=True, crop_size={"height": 512, "width": 512},
                                                    do_rescale=True, rescale_factor=0.00392156862745098, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                    image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True, )
        
        spi_tokens = ['<global>', '<local>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)
        
        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device=device)

        begin_str = "<image>\nThis provides an overview of the image.\n Please answer the following questions about the provided region. Note: Distortions include: blur, colorfulness, compression, contrast exposure and noise.\n Here is the region <global><local>. "
        
        instruction = {
            'distortion analysis': 'Provide the distortion type of this region.',
            'quality score': 'Analyze the quality of this region.',
            'importance score': 'Consider the impact of this region on the overall image quality. Analyze its importance to the overall image quality.'
        }
        
        self.ids_input = {}
        for ins_type, ins in instruction.items():
            conv = conv_templates['seagull_v1'].copy()
            qs = begin_str + ins
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            self.ids_input[ins_type] = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
    def init_image(self, img):
        if isinstance(img, dict):
            img = img['image']
        elif isinstance(img, List):
            img = cv2.imread(img[0])
            img = img[:, :, ::-1]
        h_, w_ = img.shape[:2]
        if h_ > 512:
            ratio = 512 / h_
            new_h, new_w = int(h_ * ratio), int(w_ * ratio)
            preprocessed_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            preprocessed_img = img.copy()

        return (preprocessed_img, preprocessed_img, preprocessed_img)

    def preprocess(self, img):
        image = self.image_processor.preprocess(img,
                                do_center_crop=False,
                                return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(512, 512),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)
        
        return image
    
    def seagull_predict(self, img, mask, instruct_type):
        image = self.preprocess(img)
        
        mask = np.array(mask, dtype=np.int)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            # Find the minimal bounding rectangle for the entire mask
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            w1 = x_max - x_min
            h1 = y_max - y_min
            
            bounding_box = (x_min, y_min, w1, h1)
        else:
            bounding_box = None
            
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask > 0.1, dtype=np.uint8)
        masks = torch.Tensor(mask).unsqueeze(0).to(self.model.device)
        
        input_ids = self.ids_input[instruct_type.lower()]
        
        x1, y1, w1, h1 = list(map(int, bounding_box))  # x y w h
        cropped_img = img[y1:y1 + h1, x1:x1 + w1]
        cropped_img = Image.fromarray(cropped_img)
        cropped_img = self.preprocess(cropped_img)
            
        with torch.inference_mode():

            self.model.orig_forward = self.model.forward
            self.model.forward = partial(self.model.orig_forward,
                                        img_metas=[None],
                                        masks=[masks.half()],
                                        cropped_img=cropped_img.unsqueeze(0)
                                        )
            output_ids = self.model.generate(
                input_ids,
                images=image.unsqueeze(0).half().to(self.model.device),
                do_sample=False,
                temperature=1,
                max_new_tokens=2048,
                use_cache=True,
                num_beams=1,
                top_k = 0, # 不进行topk
                top_p = 1, # 累计概率为
                )

            self.model.forward = self.model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                            skip_special_tokens=True)[0]
    
        outputs = outputs.strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[:-len(self.stop_str)]
        outputs = outputs.strip()
        if ':' in outputs:
            outputs = outputs.split(':')[1]

        outputs_list = outputs.split('.')
        outputs_list_final = []
        outputs_str = ''
        for output in outputs_list:
            if output not in outputs_list_final:
                if output=='':
                    continue
                outputs_list_final.append(output)
                outputs_str+=output+'.'
            else:
                break
        return outputs_str