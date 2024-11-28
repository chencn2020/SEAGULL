
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
- ✅ **Nov. 25, 2024**: We make SEAGULL-100w publicly available at [Hugging Face](https://huggingface.co/datasets/Zevin2023/SEAGULL-100w) and [Baidu Network](https://pan.baidu.com/s/1PY_EqdwY1FsCVfNpEXrlHA?pwd=i7h1). More details can be found at [Hugging Face](https://huggingface.co/datasets/Zevin2023/SEAGULL-100w).
- ✅ **Nov. 12, 2024**: We create this repository.


## TODO List 📝
- [x] Release the SEAGULL-100w dataset.
- [] Release the online demo.
- [] Release the checkpoints and inference codes.
- [] Release the training codes.
- [] Release the SEAGULL-3k dataset.

## Contents 📌
1. [Introduction 👀](#Introduction)
2. [Try Our Demo 🕹️](#Try-Our-Demo)
3. [Demonstrate 🎥](#Demonstrate)
4. [Acknowledgement 💌](#Acknowledgement)

## Introduction 👀
<div id="Introduction"></div>

> Existing Image Quality Assessment (IQA) methods achieve remarkable success in analyzing quality for overall image, but few works explore quality analysis for Regions of Interest (ROIs). The quality analysis of ROIs can provide fine-grained guidance for image quality improvement and is crucial for scenarios focusing on region-level quality. This paper proposes a novel network, SEAGULL, which can SEe and Assess ROIs quality with GUidance from a Large vision-Language model. SEAGULL incorporates a vision-language model (VLM), masks generated by Segment Anything Model (SAM) to specify ROIs, and a meticulously designed Mask-based Feature Extractor (MFE) to extract global and local tokens for specified ROIs, enabling accurate fine-grained IQA for ROIs. Moreover, this paper constructs two ROI-based IQA datasets, SEAGULL-100w and SEAGULL-3k, for training and evaluating ROI-based IQA. SEAGULL-100w comprises about 100w synthetic distortion images with 33 million ROIs for pre-training to improve the model's ability of regional quality perception, and SEAGULL-3k contains about 3k authentic distortion ROIs to enhance the model's ability to perceive real world distortions. After pre-training on SEAGULL-100w and fine-tuning on SEAGULL-3k, SEAGULL shows remarkable performance on fine-grained ROI quality assessment.

<img src="./imgs/SEAGULL/framework.png" alt="The framework of SEAGULL" style="height: auto; width: 100%;">

## Try Our Demo 🕹️
<div id="Try-Our-Demo"></div>

### Online demo
Click 👇 to try our demo.

<a href="https://huggingface.co/spaces/Zevin2023/SEAGULL"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces" style="max-width: 100%; height: auto;"></a>

### Offline demo

💻 **requirments:** For this demo, it needs about `17GB` GPU memory for SEAGULL(15GB) and SAM(2GB).
1. Create the environment
```
conda create -n seagull python=3.10
conda activate seagull
pip install -e .
```

2. Install [Gradio Extention](https://github.com/chencn2020/gradio-bbox) for drawing boxes on images.

3. Install Segment Anything Model.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Then download [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and put it into the ```checkpoints``` folder. 

4. Run demo on your device.
```
python app.py --model Zevin2023/SEAGULL-7B
```
> [!TIP]
> If Hugging Face is not accessible to you, try the following command.

```
HF_ENDPOINT=https://hf-mirror.com python app.py --model Zevin2023/SEAGULL-7B 
```

5. You can also download checkpoints and put them into the ```checkpoints``` folder. 

- [SEAGULL-7b](https://huggingface.co/Zevin2023/SEAGULL-7B) 
- [CLIP-convnext](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blob/main/open_clip_pytorch_model.bin)

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

Remember to change the ```"mm_vision_tower": "./checkpoints/open_clip_pytorch_model.bin"``` in ```config.json```.

Then run the following command:
```
python app.py --model ./checkpoints/SEAGULL-7B 
```

## Demonstrate 🎥
<div id="Demonstrate"></div>

<img src="./imgs/Samples/visual.png" alt="The framework of SEAGULL" style="height: auto; width: 100%;">


## Acknowledgement 💌
<div id="Acknowledgement"></div>
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