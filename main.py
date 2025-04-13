import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torchvision
print(f"Torch ver:{torch.__version__}")
print(f"TorchVision ver:{torchvision.__version__}")

from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular import data_setup,engine
from helper_functions import download_data

from helper_module.patching_visualizer import patcher_visual

#set up device for gpu
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using:{device}")

# Download pizza, steak, sushi images from GitHub
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

##args
train_dir = image_path / "train"
test_dir = image_path / "test"
batch_size = 32
NumberOfWorker =1
IMG_SIZE = 244 #from vit paper

#resize data incoming
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

print(f"Path Train:{train_dir}\nPath Test:{test_dir}")
print(f"Print Out Manually Created Transforms:{manual_transforms}")


#get data from helper modular
train_dataloader,test_dataloader,classes_name = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=batch_size,
    pin_memory=False,
    num_workers=0
    )

print(f"Training data size: {len(test_dataloader)}\nTesting data size: {len(test_dataloader)}\nclasses Name: {classes_name}")

# Get a single batch using next() with iter()
image_batch, label_batch = next(iter(train_dataloader))

# Do something with the batch
print(f"Image batch shape: {image_batch.shape}")
print(f"Label batch shape: {label_batch.shape}")

#getting image by index
image,label = image_batch[0],label_batch[0]


#replicating VIT

#crete patch number visualization
image_dim = image_batch.shape
patcher_visual(image=image,
               img_size=IMG_SIZE,
               patch_size=16,
               classes_name=classes_name,
               label=label)
