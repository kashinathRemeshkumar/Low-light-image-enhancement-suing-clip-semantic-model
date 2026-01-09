# Low-Light Image Enhancement Using CLIP Semantic Model

A deep learning approach for enhancing images captured in low-light conditions using CLIP (Contrastive Language-Image Pre-training) semantic guidance.

## Overview

This project leverages the semantic understanding capabilities of OpenAI's CLIP model to guide the enhancement of low-light images. By incorporating multimodal visual-language information, the method preserves semantic content while improving brightness, contrast, and reducing noise.

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

## Requirements
- python>=3.8
- torch>=1.10.0
- torchvision>=0.11.0
- clip-by-openai
- numpy
- opencv-python
- pillow
- matplotlib

## Example usage
from enhance import CLIPEnhancer

## Initialize the enhancer
enhancer = CLIPEnhancer(model_path='path/to/weights')

## Enhance a single image
enhanced_image = enhancer.enhance('path/to/low_light_image.jpg')

## Save the result
enhancer.save(enhanced_image, 'output/enhanced_image.jpg')

