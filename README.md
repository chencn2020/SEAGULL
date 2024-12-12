
  <img src="./imgs/Logo/logo.png" alt="SEAGULL" style="height: auto; width: 100%; margin-bottom: 3%;">

  <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; width: 100%;">
    <a href="https://huggingface.co/spaces/Zevin2023/SEAGULL"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces" style="max-width: 100%; height: auto;"></a>
    <a href="https://arxiv.org/abs/2411.10161"><img src="https://img.shields.io/badge/Arxiv-2411:10161-red" style="max-width: 100%; height: auto;"></a>
    <a href="https://huggingface.co/datasets/Zevin2023/SEAGULL-100w"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SEAGULL--100w-green" style="max-width: 100%; height: auto;"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchencn2020%2FSeagull&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors&edge_flat=false" style="max-width: 100%; height: auto;"></a>
    <a href='https://github.com/chencn2020/Seagull/stargazers'><img src='https://img.shields.io/github/stars/chencn2020/Seagull.svg?style=social' style="max-width: 100%; height: auto;"></a>
  </div>

:rocket:  :rocket: :rocket: **News:**
- To be updated...
- ✅ **Dec. 12, 2024**: We release the checkpoints [SEAGULL-7B](https://huggingface.co/Zevin2023/SEAGULL-7B) and inference codes.
- ✅ **Nov. 29, 2024**: We release the [online](#online-demo) and [offline](#offline-demo) demo for SEAGULL.
- ✅ **Nov. 25, 2024**: We make SEAGULL-100w publicly available at [Hugging Face](https://huggingface.co/datasets/Zevin2023/SEAGULL-100w) and [Baidu Netdisk](https://pan.baidu.com/s/1PY_EqdwY1FsCVfNpEXrlHA?pwd=i7h1). More details can be found at [Hugging Face](https://huggingface.co/datasets/Zevin2023/SEAGULL-100w).
- ✅ **Nov. 12, 2024**: We create this repository.


## TODO List 📝
- [x] Release the SEAGULL-100w dataset.
- [x] Release the online and offline demo.
- [x] Release the checkpoints and inference codes.
- [] Release the training codes.
- [] Release the SEAGULL-3k dataset.

## Contents 📌
1. [Introduction 👀](#Introduction)
2. [Try Our Demo 🕹️](#Try-Our-Demo)
3. [Run SEAGULL 🛠️](#Run-SEAGULL)
4. [Demonstrate 🎥](#Demonstrate)
5. [Acknowledgement 💌](#Acknowledgement)

<div id="Introduction"></div>

## Introduction 👀

<b style="color:rgb(140, 27, 19)"> TL;DR: </b> We propose a novel network (SEAGULL) and construct two datasets (SEAGULL-100w and SEAGULL-3k) to achieve fine-grained IQA for any ROIs.

> Existing Image Quality Assessment (IQA) methods achieve remarkable success in analyzing quality for overall image, but few works explore quality analysis for Regions of Interest (ROIs). The quality analysis of ROIs can provide fine-grained guidance for image quality improvement and is crucial for scenarios focusing on region-level quality. This paper proposes a novel network, SEAGULL, which can SEe and Assess ROIs quality with GUidance from a Large vision-Language model. SEAGULL incorporates a vision-language model (VLM), masks generated by Segment Anything Model (SAM) to specify ROIs, and a meticulously designed Mask-based Feature Extractor (MFE) to extract global and local tokens for specified ROIs, enabling accurate fine-grained IQA for ROIs. Moreover, this paper constructs two ROI-based IQA datasets, SEAGULL-100w and SEAGULL-3k, for training and evaluating ROI-based IQA. SEAGULL-100w comprises about 100w synthetic distortion images with 33 million ROIs for pre-training to improve the model's ability of regional quality perception, and SEAGULL-3k contains about 3k authentic distortion ROIs to enhance the model's ability to perceive real world distortions. After pre-training on SEAGULL-100w and fine-tuning on SEAGULL-3k, SEAGULL shows remarkable performance on fine-grained ROI quality assessment.

<img src="./imgs/SEAGULL/framework.png" alt="The framework of SEAGULL" style="height: auto; width: 100%;">

<div id="Try-Our-Demo"></div>

## Try Our Demo 🕹️

### Online demo
Click 👇 to try our demo. You might need to click ```Restart this Space``` to wake our demo up 🌞

<a href="https://huggingface.co/spaces/Zevin2023/SEAGULL"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces" style="max-width: 100%; height: auto;"></a>

<img src="./imgs/SEAGULL/point_demo.gif" />

<img src="./imgs/SEAGULL/mask_demo.gif" />

### Offline demo

⚠️ Please make sure the GPU memory of your device is larger than `17GB`.

1. Create the environment
```
conda create -n seagull python=3.10 -y
conda activate seagull
pip install -e .
```

2. Install [Gradio Extention](https://github.com/chencn2020/gradio-bbox) for drawing boxes on images.

> [!TIP]
> If the network is not accessible, try the following steps. It might work.

- Download the ```gradio.zip``` from [Baidu Netdisk](https://pan.baidu.com/s/1N4IuamNPpWRgaWoTndNoDA?pwd=pkyn) or [Google Drive](https://drive.google.com/file/d/1gydVOQ_OPnNMugAqojUW_-bCGpDzlC22/view?usp=sharing).
- Run ```unzip gradio.zip``` and then gain the ```gradio-bbox``` folder.
- Run ```cd gradio-bbox && pip install -e .```

3. Install Segment Anything Model.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

4. Download [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and [CLIP-convnext](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin), then put them into the ```checkpoints``` folder. 

5. Run demo on your device.
```
python app.py --model Zevin2023/SEAGULL-7B
```
> [!TIP]
> If Hugging Face is not accessible to you, try the following command.

```
HF_ENDPOINT=https://hf-mirror.com python app.py --model Zevin2023/SEAGULL-7B 
```

6. You can also download [SEAGULL-7B](https://huggingface.co/Zevin2023/SEAGULL-7B) and put them into the ```checkpoints``` folder. 

The folder structure should be:
```
├── checkpoints
    ├── SEAGULL-7B
    │   ├── config.json
    │   ├── pytorch_model-xxxxx-of-xxxxx.bin
    │   └── xxx
    ├── sam_vit_b_01ec64.pth 
    └── open_clip_pytorch_model.bin
```

Then run the following command:
```
python app.py --model ./checkpoints/SEAGULL-7B 
```

<div id="Run-SEAGULL"></div>

## Run SEAGULL 🛠️

### Preparation

1. Create the environment
```
conda create -n seagull python=3.10 -y
conda activate seagull
pip install -e .
```

2. If you want to **train** SEAGULL by yourself, install the additional packages.

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

3. Download [CLIP-convnext](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin) and [SEAGULL-7B](https://huggingface.co/Zevin2023/SEAGULL-7B), then put them into the ```checkpoints``` folder. 

The folder structure should be:
```
├── checkpoints
    ├── SEAGULL-7B
    │   ├── config.json
    │   ├── pytorch_model-xxxxx-of-xxxxx.bin
    │   └── xxx
    └── open_clip_pytorch_model.bin
```
### Usage for training

Coming soon.

### Usage for inference

1. We provide a `./demo/inference_demo.json` template for better understanding.
2. Run the following command for inference:
``` bash
python3 inference.py \
--img_dir ./imgs/Examples \
--json_path ./demo/inference_demo.json \
--mask_type rel \ # rel or points (x, y, w, h)
--inst_type quality \ # quality, importance, distortion
--model ./checkpoints/SEAGULL-7B 
```

<div id="Demonstrate"></div>

## Demonstrate 🎥

<img src="./imgs/SEAGULL/visual.png" alt="Demonstration of SEAGULL" style="height: auto; width: 100%;">

<div id="Acknowledgement"></div>

## Acknowledgement 💌

- [Osprey](https://github.com/CircleRadon/Osprey) and [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA): We build this repostory based on them.
- [RAISE](http://loki.disi.unitn.it/RAISE/): The Dist. images in SEAGULL-100w are constructed based on this dataset.
- [SAM](https://segment-anything.com/) and [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once): The mask-based ROIs are generated using these two awesome works. And SAM are used to get the segmentation result in the demo.
- [TOPIQ](https://github.com/chaofengc/IQA-PyTorch): The quality scores and importance scores for ROIs are generated using this great FR-IQA.


## Stars ⭐️

<a href="https://star-history.com/#chencn2020/Seagull&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=chencn2020/Seagull&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=chencn2020/Seagull&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=chencn2020/Seagull&type=Date" />
 </picture>
</a>

## Citation 🖊️
If our work is useful to your research, we will be grateful for you to cite our paper:
```
@misc{chen2024seagull,
      title={SEAGULL: No-reference Image Quality Assessment for Regions of Interest via Vision-Language Instruction Tuning}, 
      author={Zewen Chen and Juan Wang and Wen Wang and Sunhan Xu and Hang Xiong and Yun Zeng and Jian Guo and Shuxun Wang and Chunfeng Yuan and Bing Li and Weiming Hu},
      year={2024},
      eprint={2411.10161},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10161}, 
}
```