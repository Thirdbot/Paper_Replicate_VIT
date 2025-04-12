import torch
import torchvision
print(f"Torch ver:{torch.__version__}")
print(f"TorchVision ver:{torchvision.__version__}")

import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular import data_setup,engine


#set up device for gpu

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using:{device}")