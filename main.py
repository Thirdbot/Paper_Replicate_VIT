import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchvision
print(f"Torch ver:{torch.__version__}")
print(f"TorchVision ver:{torchvision.__version__}")

from torch import nn
import torch.nn.functional as f
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
                 batch_size:int=32
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
        self.batch_size = batch_size
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        x_patched= self.patcher(x) #batchsize,channel,height,width
        x_flattened = self.flatten(x_patched) #batchsize,embedding_dim,num_patches
        x_flattened_transpose = x_flattened.permute(0,2,1) #batchsize,num_patches,embedding_dim
        padding=torch.zeros((self.batch_size-x_flattened_transpose.shape[0],x_flattened_transpose.shape[1],x_flattened_transpose.shape[2]))        
        x_flattened_transpose = f.pad(x_flattened_transpose,pad=(0,0,0,0,0,padding.shape[0]))
        # print(f"X flattened transpose shape:{x_flattened_transpose.shape}")
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
                 embedding_dim:int,
                 batch_size:int=32
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.class_token = nn.Parameter(torch.randn(self.batch_size,1,embedding_dim))
        
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        # padding=torch.zeros((x.shape[0],x.shape[1],self.embedding_dim))        
        # x = f.pad(x,pad=(0,0,0,0,0,padding.shape[0]))
        
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
        x_patched = self.patch_embedding(x)
        # print(f"X patched shape:{x_patched.shape}")
        
        class_token = ClassTokenEmbedding(embedding_dim=self.embedding_dim,batch_size=x_patched.shape[0])
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
        
        # Move to device
        self.to(device)
        
    def forward(self,x):
        x = x.to(device)
        x = self.multi_head_attention(x) + x
       
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
                 mlp_size:int=768,
                 dropout:float=0.1,
                 num_transformer_layers:int=12,
                 num_classes:int=1000,
                 img_size:int=244,
                 patch_size:int=16,
                 hidden_dim:int=768,
                 batch_size:int=32
                 ):
        super().__init__()
        self.batch_size = batch_size
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
        self.mlp_layer = nn.Sequential(*[MLPBlock(embedding_dim=self.embedding_dim,
                                 mlp_size=self.mlp_size,
                                 dropout=self.dropout)
                                     for _ in range(self.num_transformer_layers)])
        
        self.hidden_dim = hidden_dim
        #weight for randomize
        self.weight = torch.randn(1,self.batch_size,self.num_patches+1,self.hidden_dim,self.embedding_dim).to(device)

        #store weight of attention x  randomize
        self.list_weight = nn.Parameter(torch.tensor([],device=device))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim,out_features=self.num_classes)
        )
        # Move to device
        self.to(device)
        
    def transform_weight(self,encoder_output,weight):
        weight = weight.permute(1,0,2,3,4)
            
        # print(f"Weight shape:{weight.shape}")
        
        #pad with zero ensure batch size is the same
        # print(f"Encoder output shape:{encoder_output.shape}")
        x = encoder_output
        # padding=torch.zeros((encoder_output.shape[0],encoder_output.shape[1],self.embedding_dim))        
        # x = f.pad(x,pad=(0,0,0,0,0,padding.shape[0]))
        
        # print(f"Padding shape:{x.shape}")
        
        # Now do the unsqueeze operations of x
        expanded_x = x.unsqueeze(0)  # Add row dimension
        unsqueeze_x = expanded_x.unsqueeze(3)  # Add patch dimension
        # unsqueeze_x = unsqueeze_x.unsqueeze(4)  # Add embedding dimension
        unsqueeze_x = unsqueeze_x.expand(1,encoder_output.shape[0],encoder_output.shape[1],self.hidden_dim,encoder_output.shape[2])# Add hidden dimensio
        # Permute to match weight dimensions
        permute_x = unsqueeze_x.permute(1,0,2,3,4)
        
        #perfrom collasp dim
        # print(f"Permute x shape:{permute_x.shape}")
        # print(f"Weight shape:{weight.shape}")
        output = torch.einsum('b a h s e, b a h s e -> a b h e', permute_x, weight)
        
        return output
    
    def forward(self,x):
        x = x.to(device)
        x = self.image_embedding(x)  # [batch_size, num_patches + 1, embedding_dim]
        encoder_output = self.transfromer_encoder_layers(x)  # [batch_size, num_patches + 1, embedding_dim]
        #seperate ml from encoder
        mlp_output = self.mlp_layer(encoder_output)
        mlp_output = mlp_output + encoder_output
        
        
        W_output = self.transform_weight(encoder_output,self.weight)
        
        #concatenate output with list_weight
        # print(f"W_output shape:{W_output.shape}")
        self.list_weight.data = torch.cat((self.list_weight.data,W_output),0)
        # print(f"List weight shape:{self.list_weight.shape}")
        
        #sum concat weight
        sum_weight = torch.sum(self.list_weight.data,dim=0)
        # sum_weight = sum_weight.square()
        self.list_weight.data = sum_weight.unsqueeze(0)
        #encoder output * sum_weight
        weighted_output = torch.einsum('b p e, e p b -> b p e', encoder_output, sum_weight.transpose(0,2))
        #mlp output(decoder) * weighted_output
        mlped_output = torch.einsum('b p e, e p b -> b p e', mlp_output, weighted_output.transpose(0,2))
        # Extract only the class token (first token) for classification
        class_token = mlped_output[:, 0, :]  # [batch_size, embedding_dim]
        
        # Pass through classifier
        x = self.classifier(class_token)  # [batch_size, num_classes]
        
    
        
        return x




# Create both models
vit = VIT(embedding_dim=embendding_dim,
          num_heads=3,
          mlp_size=128,
          hidden_dim=128,
          dropout=0.1,
          num_transformer_layers=6,
          num_classes=len(classes_name),
          img_size=IMG_SIZE,
          patch_size=16)

# vit_output,vit_encoder_output = vit(image_batch)
# print(f"VIT shape:{vit_output.shape}")
# print(f"VIT encoder output:{vit_encoder_output}")
# print(f"VIT encoder output shape:{vit_encoder_output.shape}")

# Create optimizer
vit_optimizer = torch.optim.Adam(params=vit.parameters(),
                            lr=0.001,
                            betas=(0.9,0.999),
                            weight_decay=0.1)
    

summary(vit,input_size=(32,3,244,244))
# summary(weight_model)
# Loss function
loss_fn = nn.CrossEntropyLoss()

# Train the combined model
results = engine.train(model1=vit,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      vit_optimizer=vit_optimizer,
                      loss_fn=loss_fn,
                      epochs=10,
                      device=device,
                      batch_size=batch_size)

plot_loss_curves(results)

