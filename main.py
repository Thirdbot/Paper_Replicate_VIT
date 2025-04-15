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
# single_image_visual(image.permute(1,2,0),classes_name,label)

#replicating VIT------------------------------------------------------

#crete patch number visualization
patch_size = 16
embendding_dim = 768

# patcher_visual(image=image,
#                img_size=IMG_SIZE,
#                patch_size=patch_size,
#                classes_name=classes_name,
#                label=label)

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

#CREATE PATCH EMBEDDING LAYERS
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768,
                 ):
        super().__init__()
        #given image shape is batchsize,channel,height,width
        #the patch size is 16 so we need to split the image into 16x16 patch
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        #current shape after flattening is collasping of last 2 dims
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)
        
    def forward(self,x):
        x_patched= self.patcher(x) #batchsize,channel,height,width
        x_flattened = self.flatten(x_patched) #batchsize,embedding_dim,num_patches
        x_flattened_transpose = x_flattened.permute(0,2,1) #batchsize,num_patches,embedding_dim
        return x_flattened_transpose


patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)

print(f"Patchify shape:{patchify(image.unsqueeze(0)).shape}")

#CREATE CLASS TOKEN EMBEDDING LAYERS
 
patch_embedding = patchify(image.unsqueeze(0))

class ClassTokenEmbedding(nn.Module):
    def __init__(self,
                 batch_size:int,
                 embedding_dim:int
                 ):
        
        super().__init__()
        
        self.class_token = nn.Parameter(torch.randn(batch_size,1,embedding_dim))
       
    
    def forward(self,x):
        image_embedded_class = torch.cat((self.class_token,x),dim=1)
        return image_embedded_class
    
class_tokenning = ClassTokenEmbedding(batch_size=patch_embedding.shape[0],
                                      embedding_dim=patch_embedding.shape[-1])

embbeding_with_class = class_tokenning(patch_embedding)
print(f"Class tokenning shape:{embbeding_with_class.shape}")

class PositionEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim:int,
                 patch_size:int):
        
        super().__init__()
        
        self.position_embedding = nn.Parameter(torch.randn(1,patch_size,embedding_dim))

    def forward(self,x):
        return x + self.position_embedding

position_embedding = PositionEmbedding(embedding_dim=embbeding_with_class.shape[-1],
                                       patch_size=embbeding_with_class.shape[1])


class_position_embedding = position_embedding(embbeding_with_class)

print(f"Class position embedding shape:{class_position_embedding.shape}")


