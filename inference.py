import argparse
from demo.seagull_inference import Seagull
import json
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SEAGULL', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--img_dir', help='path to images', default='./imgs/Examples')
    parser.add_argument('--json_path', help='path to json with rois', default='./demo/inference_demo.json')
    parser.add_argument('--mask_type', help='types to indicate the rois. Rel: Mask-based ROIs, Points: BBox-based ROIs', choices=['rle', 'points'], default='points')
    parser.add_argument('--inst_type', help='the instruction to SEAGULL', choices=['quality', 'importance', 'distortion'], default='distortion')
    parser.add_argument('--model', help='path to seagull model', default='Zevin2023/SEAGULL-7B')
    args = parser.parse_args()
    
    SEAGULL = Seagull(args.model) # load model
    
    with open(args.json_path, 'r') as f:
        inference_samples = json.load(f)
    
    for inference_sample in inference_samples:
        img_file = inference_sample['img']
        mask_info = inference_sample.get(args.mask_type, None)
        
        assert mask_info is not None, 'Make sure the mask labels in json are correct.'
        
        # get predicted results
        res = SEAGULL.seagull_predict(os.path.join(args.img_dir, img_file), mask_info, instruct_type=args.inst_type, mask_type=args.mask_type)
        print(img_file, res)
    
    
    
    