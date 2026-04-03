# Low-Light Image Enhancement Using CLIP Semantic Model

A deep learning approach for enhancing images captured in low-light conditions using CLIP (Contrastive Language-Image Pre-training) semantic guidance.

## Overview

This project leverages the semantic understanding capabilities of OpenAI's CLIP model to guide the enhancement of low-light images. By incorporating multimodal visual-language information, the method preserves semantic content while improving brightness, contrast, and reducing noise.

## Sample Output

<p align="center">
  <img src="output.png" alt="Low-light Input vs Enhanced Output" width="90%"/>
  <br/>
  <em>Left: Low-light input &nbsp;|&nbsp; Right: Enhanced output</em>
</p>

## Features

- **Semantic-Aware Enhancement**: Uses CLIP's semantic understanding to maintain image content integrity
- **Zero-Shot Capability**: Works without requiring paired normal-light/low-light training datasets
- **Noise Reduction**: Effectively suppresses noise while enhancing illumination
- **Structure Preservation**: Maintains spatial and semantic structures during enhancement

## Technical Approach

The method combines:
- CLIP pre-trained model for semantic feature extraction
- Deep learning enhancement network guided by semantic information
- Loss functions that balance illumination enhancement and semantic consistency

## Quantitative Results on LOL Dataset

| Type | Model | Reference | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|------|-------|-----------|--------|--------|---------|-------|
| Unpaired Training | EnlightenGAN | TIP'21 | 17.873 | 0.653 | 0.395 | 102.879 |
| Unpaired Training | CLIP-Lit | ICCV'23 | 12.714 | 0.481 | 0.363 | 120.100 |
| Unpaired Training | NeRco | ICCV'23 | 22.946 | 0.773 | 0.527 | 27.095 |
| Unpaired Training | PairLIE | CVPR'23 | 19.735 | 0.776 | 0.307 | 98.050 |
| Unpaired Training | LightenDiffusion | ECCV'24 | 21.099 | 0.829 | 0.310 | 93.784 |
| Zero-Shot | Zero-DCE | CVPR'20 | 15.053 | 0.582 | 0.602 | 96.571 |
| Zero-Shot | Zero-DCE++ | TPAMI'21 | 14.682 | 0.622 | 0.507 | 85.552 |
| Zero-Shot | RUAS | CVPR'21 | 16.504 | 0.488 | 0.375 | 116.757 |
| Zero-Shot | SCI | CVPR'22 | 14.651 | 0.502 | 0.302 | 81.456 |
| Zero-Shot | GDP | CVPR'24 | 15.896 | 0.402 | 0.572 | 123.362 |
| Zero-Shot | FourierDiff | CVPR'24 | 18.673 | 0.602 | 0.376 | 76.395 |
| Zero-Shot | **Ours** | - | **15.5568** | **0.792** | **0.326** | 133.2763 |

> ↑ Higher is better &nbsp;|&nbsp; ↓ Lower is better

## Requirements

python>=3.8

torch>=1.10.0

torchvision>=0.11.0

clip-by-openai

numpy

opencv-python

pillow

matplotlib

## Example Usage

python
from enhance import CLIPEnhancer

# Initialize the enhancer
enhancer = CLIPEnhancer(model_path='path/to/weights')

# Enhance a single image
enhanced_image = enhancer.enhance('path/to/low_light_image.jpg')

# Save the result
enhancer.save(enhanced_image, 'output/enhanced_image.jpg')
