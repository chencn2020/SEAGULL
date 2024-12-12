import cv2
from PIL import Image
import numpy as np
import torch
import gradio as gr

def process_mask_to_show(mask):
    '''
        Process the mask to show on the gradio.Image
    '''
    mask = np.array(mask > 0.1, dtype=np.uint8) * 255
    mask_stacked = np.stack([mask] * 3, axis=-1)

    return mask_stacked

def img_add_masks(img_, colored_mask, mask, linewidth=2):
    if type(img_) is np.ndarray:
        img = Image.fromarray(img_, mode='RGB').convert('RGBA')
    else:
        img = img_.copy()
    h, w = img.height, img.width
    # contour
    temp = np.zeros((h, w, 1))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(temp, contours, -1, (255, 255, 255), linewidth)
    color = np.array([1, 1, 1, 1])
    contour_mask = temp * color.reshape(1, 1, -1)

    overlay_inner = Image.fromarray(colored_mask.astype(np.uint8), 'RGBA')
    img.paste(overlay_inner, (0, 0), overlay_inner)

    overlay_contour = Image.fromarray(contour_mask.astype(np.uint8), 'RGBA')
    img.paste(overlay_contour, (0, 0), overlay_contour)
    return img

def gen_colored_masks(
        annotation,
        random_color=False,
):
    """
    Code is largely based on https://github.com/CASIA-IVA-Lab/FastSAM/blob/4d153e909f0ad9c8ecd7632566e5a24e21cf0071/utils/tools_gradio.py#L130
    """
    device = annotation.device
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]

    index = (annotation != 0).to(torch.long).argmax(dim=0)
    if random_color:
        color = torch.rand((mask_sum, 1, 1, 3)).to(device)
    else:
        color = torch.ones((mask_sum, 1, 1, 3)).to(device) * torch.tensor(
            [30 / 255, 144 / 255, 255 / 255]
        ).to(device)
    transparency = torch.ones((mask_sum, 1, 1, 1)).to(device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual

    mask = torch.zeros((height, weight, 4)).to(device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight))
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]
    mask_cpu = mask.cpu().numpy()

    return mask_cpu, sorted_indices

def mask_foreground(mask, trans=0.6, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3) * 255, np.array([trans * 255])], axis=0)
    else:
        color = np.array([30, 144, 255, trans * 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    return mask_image


def mask_background(mask, trans=0.5):
    h, w = mask.shape[-2:]
    mask_image = (1 - mask.reshape(h, w, 1)) * np.array([0, 0, 0, trans * 255])

    return mask_image


def mask_select_point(all_masks, output_mask_2_raw, mask_order, evt: gr.SelectData):
    h, w = output_mask_2_raw.height, output_mask_2_raw.width
    pointed_mask = None
    for i in range(len(mask_order)):
        idx = mask_order[i]
        msk = all_masks[idx]
        if msk[evt.index[1], evt.index[0]] == 1:
            pointed_mask = msk.copy()
            break

    if pointed_mask is not None:
        contours, hierarchy = cv2.findContours(pointed_mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ret = output_mask_2_raw.copy()

        temp = np.zeros((h, w, 1))
        contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(temp, contours, -1, (255, 255, 255), 3)
        color = np.array([1, 1, 1, 1])
        contour_mask = temp * color.reshape(1, 1, -1)

        colored_mask = mask_background(pointed_mask)

        overlay_inner = Image.fromarray(colored_mask.astype(np.uint8), 'RGBA')
        ret.paste(overlay_inner, (0, 0), overlay_inner)

        overlay_contour = Image.fromarray(contour_mask.astype(np.uint8), 'RGBA')
        ret.paste(overlay_contour, (0, 0), overlay_contour)

        return ret, pointed_mask
    else:
        return output_mask_2_raw, None
