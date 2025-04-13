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

from helper_module.grid_visual import patcher_visual
from helper_module.single_visual import single_image_visual
from helper_module.grid_visual import grid_index_visual

import matplotlib.pyplot as plt
import random

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

#visualization
# single_image_visual(image,classes_name,label)





#replicating VIT




#crete patch number visualization
patch_size = 16

# patcher_visual(image=image,
#                img_size=IMG_SIZE,
#                patch_size=patch_size,
#                classes_name=classes_name,
#                label=label)


##create conv2d layer

#since we already have split patch image to different patch so stride going to be patch size
conv2d = nn.Conv2d(in_channels=3,
                   out_channels=768,
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)

#image of conv2d from full image
image_of_conv2d = conv2d(image.unsqueeze(0)) #add batch dimension batchsize,embedd_dim,P,P
print(f"Image of conv2d shape:{image_of_conv2d.shape}")

#visualize embbeded patches
# r=2
# c=20
# random_indexs = random.sample(range(0,768),k=c)  # Changed from 758 to 768 to match out_channels
# box_size = 12

# print(f"Random indices: {random_indexs}")
# print(f"Number of indices: {len(random_indexs)}")

# fig,axs = plt.subplots(nrows=r,ncols=c,figsize=(box_size,box_size))


# print(f"Image for visualization shape: {image_of_conv2d.shape}")

# grid_index_visual(image=image_of_conv2d,
#             fig=fig,
#             axs=axs,
#             index_iel=[0],
#             index_jel=random_indexs,
#             classes_name=classes_name,
#             label=label)

flatten_image_of_conv2d = torch.flatten(image_of_conv2d,start_dim=2,end_dim=3)
print(f"Flatten image of `conv2d` shape:{flatten_image_of_conv2d.shape}")
flatten_image_of_conv2d_transpose = flatten_image_of_conv2d.permute(0,2,1)
print(f"Flatten image of `conv2d` transpose shape:{flatten_image_of_conv2d_transpose.shape}")











