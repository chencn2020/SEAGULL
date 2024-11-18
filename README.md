
  <img src="./imgs/Logo/logo.png" alt="SEAGULL" style="height: auto; width: 100%; margin-bottom: 3%;">

  <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; width: 100%;">
    <a href=""><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces" style="max-width: 100%; height: auto;"></a>
    <a href="https://arxiv.org/abs/2411.10161"><img src="https://img.shields.io/badge/Arxiv-2411:10161-red" style="max-width: 100%; height: auto;"></a>
    <a href=""><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green" style="max-width: 100%; height: auto;"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchencn2020%2FSeagull&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors&edge_flat=false" style="max-width: 100%; height: auto;"></a>
    <a href='https://github.com/chencn2020/Seagull/stargazers'><img src='https://img.shields.io/github/stars/chencn2020/Seagull.svg?style=social' style="max-width: 100%; height: auto;"></a>
  </div>

:rocket:  :rocket: :rocket: **News:**
- To be updated...
- ✅ **Nov. 12, 2024**: We created this repository.


## TODO List 📝
- [] Release the checkpoints and inference codes.
- [] Release the training codes.
- [] Release the Seagull-100w and Seagull-3k datasets.
- [] Release the online demo

## Contents
1. [Introduction](#Introduction)
2. [Demonstrate](#Demonstrate)

## Introduction
<div id="Introduction"></div>

> Existing Image Quality Assessment (IQA) methods achieve remarkable success in analyzing quality for overall image, but few works explore quality analysis for Regions of Interest (ROIs). The quality analysis of ROIs can provide fine-grained guidance for image quality improvement and is crucial for scenarios focusing on region-level quality. This paper proposes a novel network, SEAGULL, which can SEe and Assess ROIs quality with GUidance from a Large vision-Language model. SEAGULL incorporates a vision-language model (VLM), masks generated by Segment Anything Model (SAM) to specify ROIs, and a meticulously designed Mask-based Feature Extractor (MFE) to extract global and local tokens for specified ROIs, enabling accurate fine-grained IQA for ROIs. Moreover, this paper constructs two ROI-based IQA datasets, SEAGULL-100w and SEAGULL-3k, for training and evaluating ROI-based IQA. SEAGULL-100w comprises about 100w synthetic distortion images with 33 million ROIs for pre-training to improve the model's ability of regional quality perception, and SEAGULL-3k contains about 3k authentic distortion ROIs to enhance the model's ability to perceive real world distortions. After pre-training on SEAGULL-100w and fine-tuning on SEAGULL-3k, SEAGULL shows remarkable performance on fine-grained ROI quality assessment.

<img src="./imgs/SEAGULL/framework.png" alt="The framework of SEAGULL" style="height: auto; width: 100%;">

## Demonstrate
<div id="Demonstrate"></div>

<img src="./imgs/Samples/visual.png" alt="The framework of SEAGULL" style="height: auto; width: 100%;">

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
@misc{chen2024seagullnoreferenceimagequality,
      title={SEAGULL: No-reference Image Quality Assessment for Regions of Interest via Vision-Language Instruction Tuning}, 
      author={Zewen Chen and Juan Wang and Wen Wang and Sunhan Xu and Hang Xiong and Yun Zeng and Jian Guo and Shuxun Wang and Chunfeng Yuan and Bing Li and Weiming Hu},
      year={2024},
      eprint={2411.10161},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10161}, 
}
```