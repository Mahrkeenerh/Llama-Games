# LLaMA Games

A deep learning project implementing image captioning and game description generation using LLaMA 3.1 8B model combined with Vision Transformer (ViT) for image understanding.

## Overview

This project implements a multimodal system that combines:
- Vision Transformer (ViT) for image feature extraction
- LLaMA 3.1 8B as the base language model
- LoRA fine-tuning for task adaptation
- A custom projection layer to bridge vision and language models
- Support for both image captioning and detailed game description generation

## Key Features

- **Multimodal Architecture**:
  - Vision Transformer (ViT) for image processing
  - LLaMA 3.1 8B with LoRA adaptation for text generation
  - Projector module to map image embeddings to language embeddings
  
- **Training Approaches**:
  - LoRA (Low-Rank Adaptation) fine-tuning only
  - Both full-precision and 8-bit quantized training options
  
- **Dataset Support**:
  - COCO dataset for image captioning pre-training
  - Custom games dataset for app descriptions
  - Support for both single and multiple image inputs

- **Evaluation Tools**:
  - BLEU, METEOR, and ROUGE score calculation
  - Semantic similarity analysis
  - Visualization tools for model outputs

## Model Architecture

The system uses a three-part architecture:

1. **Vision Encoder**: 
   - ViT for processing input images
   - Produces image embeddings in the vision space

2. **Projection Layer**: 
   - Custom layer that maps vision embeddings to language embedding space
   - Handles dimensionality matching between vision and language models
   - Supports processing multiple images with configurable merging strategies

3. **Language Model**: 
   - Base model: LLaMA 3.1 8B
   - LoRA adaptation for task-specific fine-tuning
   - Options for both full-precision and 8-bit quantized operation

Key features:
- Multiple image support with configurable merging strategies
- Efficient LoRA fine-tuning approach preserving base model weights
- Optional 8-bit quantization for reduced memory footprint
- Special token handling for task-specific prompting

## Evaluation

The project includes multiple evaluation methods:

- **Automated Metrics**:
  - BLEU score for n-gram precision
  - METEOR score for semantic matching
  - ROUGE score for recall evaluation

- **Semantic Analysis**:
  - Sentence transformer-based similarity scoring
  - Cross-caption similarity evaluation
  - Intra-caption consistency checks

- **Visual Analysis**:
  - Interactive notebooks for result visualization
  - Training progression analysis
  - Loss and learning rate tracking

## Acknowledgments

- Meta for the LLaMA 3.1 model
- Hugging Face for transformers library
- NLP Connect for the initial ViT-GPT2 image captioning model
