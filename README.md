# PyTorch Deep Learning Project

This project is part of my learning journey in deep learning with PyTorch, following the excellent resources and tutorials from Daniel Bourke (mrdbourke).

## Acknowledgments

- The `helper_functions.py` file is sourced from [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py)
- Special thanks to Daniel Bourke for his comprehensive PyTorch tutorials and resources

## Project Structure

```
.
├── main.py              # Main project file
├── helper_functions.py  # Utility functions for PyTorch (from mrdbourke)
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Dependencies

Core Dependencies:
- PyTorch 2.6.0 (CUDA 12.6) - from PyPI
- torchvision 0.21.0 (CUDA 12.6) - from PyPI
- torchaudio 2.6.0 (CUDA 12.6) - from PyPI
- torchinfo 1.8.0 - from conda-forge
- matplotlib 3.10.0
- numpy 2.1.2

## Setup

1. Clone this repository
2. Create and set up your environment:

   ```bash
   # Create a new conda environment
   conda create -n pytorch_env python=3.12
   conda activate pytorch_env

   # Install PyTorch with CUDA support from PyPI
   # Note: Install torch first, then torchvision and torchaudio
   pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
   pip install torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

   # Install torchinfo from conda-forge
   conda install -c conda-forge torchinfo

   # Install additional dependencies
   pip install matplotlib numpy
   ```

   Note: We use PyPI for PyTorch packages to ensure CUDA compatibility and conda-forge for torchinfo to avoid conflicts.

## Usage

1. Make sure your conda environment is activated:
   ```bash
   conda activate pytorch_env
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Troubleshooting

If you encounter the error:
```
ModuleNotFoundError: No module named 'torch.hub'
```

Try these solutions:

1. Remove all PyTorch-related packages and reinstall in the correct order:
   ```bash
   conda remove torchinfo --force
   pip uninstall torch torchvision torchaudio
   
   # Reinstall in correct order
   pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
   pip install torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
   conda install -c conda-forge torchinfo
   ```

2. If the error persists, try installing PyTorch without CUDA first, then upgrade:
   ```bash
   pip install torch torchvision torchaudio
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

## Learning Resources

- [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
- [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)

## License

This project is for educational purposes. The `helper_functions.py` file is used under the original repository's license terms.
