# PyTorch Deep Learning Project (Experimental Vision Transformer)

This project is part of my learning journey in deep learning with PyTorch, following the excellent resources and tutorials from Daniel Bourke (mrdbourke). This version includes an experimental implementation of the Vision Transformer (ViT) architecture.

## Project Overview

This experimental version implements a custom Vision Transformer (ViT) architecture with the following components:
- Patch Embedding
- Class Token Embedding
- Position Embedding
- Multi-Head Attention
- MLP Blocks
- Transformer Encoder Layers

## Acknowledgments

- The `helper_functions.py` file is sourced from [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py)
- Special thanks to Daniel Bourke for his comprehensive PyTorch tutorials and resources
- Based on the original Vision Transformer paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

## Project Structure

```
.
├── main.py              # Main project file with experimental ViT implementation
├── helper_functions.py  # Utility functions for PyTorch (from mrdbourke)
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Dependencies

Core Dependencies:
- PyTorch 2.6.0 (CUDA 12.6)
- torchvision 0.21.0
- torchaudio 2.6.0
- torchinfo 1.8.0
- matplotlib 3.10.0
- numpy 2.1.2

## Setup

1. Clone this repository
2. Create and set up your environment:

   ```bash
   # Create a new conda environment
   conda create -n pytorch_env python=3.12
   conda activate pytorch_env

   # Install all dependencies
   pip install -r requirements.txt
   ```

## Usage

1. Make sure your conda environment is activated:
   ```bash
   conda activate pytorch_env
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Experimental Features

- Custom implementation of Vision Transformer architecture
- Modular design with separate components for patch embedding, attention, and classification
- Support for different image sizes and patch configurations
- Configurable number of transformer layers and attention heads

## Learning Resources

- [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
- [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

## License

This project is for educational purposes. The `helper_functions.py` file is used under the original repository's license terms.
