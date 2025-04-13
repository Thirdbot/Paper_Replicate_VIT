import matplotlib.pyplot as plt


def single_image_visual(image,classes,label):
    plt.imshow(image)
    plt.title(classes[label])
    plt.axis(False)
    plt.show()