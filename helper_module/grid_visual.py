import matplotlib.pyplot as plt


def patcher_visual(image, img_size: int, patch_size: int, classes_name, label):
    image_permuted = image.permute(1, 2, 0)
    num_patches = img_size//patch_size
    fig, axs = plt.subplots(nrows=num_patches,
                           ncols=num_patches,
                           figsize=(num_patches, num_patches),
                           sharex=True,
                           sharey=True)
    
    for i in range(num_patches):
        for j in range(num_patches):
            patch_height = i * patch_size
            patch_width = j * patch_size
            axs[i,j].imshow(image_permuted[patch_height:patch_height+patch_size,
                           patch_width:patch_width+patch_size,
                           :])
            axs[i,j].set_ylabel(i+1,
                               rotation="horizontal",
                               horizontalalignment="right",
                               verticalalignment="center")
            axs[i,j].set_xlabel(j+1)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].label_outer()
    
    fig.suptitle(f"{classes_name[label]} -> Patchified", fontsize=14)
    plt.show()

def grid_index_visual(image,fig=None,axs=None,index_iel=[],index_jel=[], classes_name=None, label=None):
    assert (fig is not None and axs is not None), "fig and axs must both be provided"
    for i,index_i in enumerate(index_iel):
        for j,index_j in enumerate(index_jel):
            image_conv2d_detach = image[index_i,index_j,:,:]
            axs[i,j].imshow(image_conv2d_detach.squeeze().detach().numpy())
            axs[i,j].axis("off")
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].label_outer()
            axs[i,j].set_xlabel(j+1)
            axs[i,j].set_ylabel(i+1,
                               rotation="horizontal",
                               horizontalalignment="right",
                               verticalalignment="center")
    fig.suptitle(f"{classes_name[label]}", fontsize=14)
    plt.show()
