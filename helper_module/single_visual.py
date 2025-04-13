import matplotlib.pyplot as plt


def single_image_visual(image,classes,label):
    plt.imshow(image.permute(1,2,0))
    plt.title(classes[label])
    plt.axis(False)
    plt.show()