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

from helper_functions import plot_loss_curves
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

#PATCH EMBEDDING
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
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        x_patched= self.patcher(x) #batchsize,channel,height,width
        x_flattened = self.flatten(x_patched) #batchsize,embedding_dim,num_patches
        x_flattened_transpose = x_flattened.permute(0,2,1) #batchsize,num_patches,embedding_dim
        return x_flattened_transpose


# patchify = PatchEmbedding(in_channels=3,
#                           patch_size=16,
#                           embedding_dim=768)

# print(f"Patchify shape:{patchify(image.unsqueeze(0)).shape}")

# #CREATE CLASS TOKEN EMBEDDING LAYERS
 
# patch_embedding = patchify(image.unsqueeze(0))

#CLASS TOKEN EMBEDDING
class ClassTokenEmbedding(nn.Module):
    def __init__(self,
                 batch_size:int,
                 embedding_dim:int
                 ):
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(batch_size,1,embedding_dim))
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        image_embedded_class = torch.cat((self.class_token,x),dim=1)
        return image_embedded_class
    
# class_tokenning = ClassTokenEmbedding(batch_size=patch_embedding.shape[0],
#                                       embedding_dim=patch_embedding.shape[-1])

# embbeding_with_class = class_tokenning(patch_embedding)
# print(f"Class tokenning shape:{embbeding_with_class.shape}")

#POSITION EMBEDDING
class PositionEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim:int,
                 patch_size:int):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1,patch_size,embedding_dim))
        # Move to device
        self.to(device)

    def forward(self,x):
        x = x.to(device)
        return x + self.position_embedding

# position_embedding = PositionEmbedding(embedding_dim=embbeding_with_class.shape[-1],
#                                        patch_size=embbeding_with_class.shape[1])


# class_position_embedding = position_embedding(embbeding_with_class)

# print(f"Class position embedding shape:{class_position_embedding.shape}")





#IMAGE EMBEDDING
class ImageEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim:int,
                 in_channels:int=3,
                 patch_size:int=16
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        self.patch_embedding = PatchEmbedding(in_channels=self.in_channels,
                                              patch_size=self.patch_size,
                                              embedding_dim=self.embedding_dim)
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        batch_size = x.shape[0]
        x_patched = self.patch_embedding(x)
        # print(f"X patched shape:{x_patched.shape}")
        
        class_token = ClassTokenEmbedding(batch_size=batch_size,
                                         embedding_dim=self.embedding_dim)
        x_class_token = class_token(x_patched)
        # print(f"X class token shape:{x_class_token.shape}")
        
        position_embedding = PositionEmbedding(embedding_dim=self.embedding_dim,
                                              patch_size=x_class_token.shape[1])
        x_position_embedding = position_embedding(x_class_token)
        return x_position_embedding


# image_embedding = ImageEmbedding(embedding_dim=embendding_dim)

# print(f"Image embedding shape:{image_embedding(image_batch).shape}")
    
            
#MULTI HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0.0):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                         num_heads=num_heads,
                                                         dropout=attn_dropout,
                                                         batch_first=True)
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        x = self.layernorm(x)
        attn_output, _ = self.multi_head_attention(query=x,
                                                 key=x,
                                                 value=x,
                                                 need_weights=False)
        return attn_output

# multi_head_attention = MultiHeadAttention(embedding_dim=embendding_dim)
#in genarally this is out put which batch from (batch_size,num_patches,embedding_dim)
#that have the most attention 
# print(f"Multi head attention shape:{multi_head_attention(image_embedding(image_batch)).shape}")

class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_size,out_features=embedding_dim))
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        x = self.layernorm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 dropout:float=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.dropout = dropout
        
        self.multi_head_attention = MultiHeadAttention(embedding_dim=self.embedding_dim,
                                                      num_heads=self.num_heads)
        self.mlp_block = MLPBlock(embedding_dim=self.embedding_dim,
                                 mlp_size=self.mlp_size,
                                 dropout=self.dropout)
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        x = self.multi_head_attention(x) + x
        x = self.mlp_block(x) + x
        return x

# transformer_encoder_block = TransformerEncoderBlock(embedding_dim=embendding_dim)

# encoded_output = transformer_encoder_block(image_embedding(image_batch))
# print(f"Transformer encoder block shape:{encoded_output.shape}")

# summary(transformer_encoder_block,input_size=encoded_output.shape,
#         col_names=["input_size","output_size","num_params","trainable"],
#         col_width=20,
#         depth=1)

class VIT(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 dropout:float=0.1,
                 num_transformer_layers:int=12,
                 num_classes:int=1000,
                 img_size:int=224,
                 patch_size:int=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_size = mlp_size
        self.dropout = dropout
        self.num_transformer_layers = num_transformer_layers
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.image_embedding = ImageEmbedding(embedding_dim=self.embedding_dim,
                                             patch_size=self.patch_size)
        
        self.transfromer_encoder_layers = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=self.embedding_dim,
                                                             num_heads=self.num_heads,
                                                             mlp_size=self.mlp_size,
                                                             dropout=self.dropout)
                                                     for _ in range(self.num_transformer_layers)])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim,out_features=self.num_classes)
        )
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        x = self.image_embedding(x)  # [batch_size, num_patches + 1, embedding_dim]
        x = self.transfromer_encoder_layers(x)  # [batch_size, num_patches + 1, embedding_dim]
        
        # Extract only the class token (first token) for classification
        class_token = x[:, 0, :]  # [batch_size, embedding_dim]
        
        # Pass through classifier
        x = self.classifier(class_token)  # [batch_size, num_classes]
        return x

vit = VIT(embedding_dim=embendding_dim,
          num_heads=12,
          mlp_size=3072,
          dropout=0.1,
          num_transformer_layers=12,
          num_classes=len(classes_name),
          img_size=224,
          patch_size=16)

print(f"VIT shape:{vit(image_batch).shape}")

# summary(vit,input_size=[1,3,244,244],
#         col_names=["input_size","output_size","num_params","trainable"],
#         col_width=20,
#         depth=1)

optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=0.001,
                             betas=(0.9,0.999),
                             weight_decay=0.1)

loss_fn = nn.CrossEntropyLoss()

results = engine.train(model=vit,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=3,
                       device=device)

plot_loss_curves(results)

