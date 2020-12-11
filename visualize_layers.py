from cnn import device
from cnn.helper import set_dataset_and_loaders
from util.tensorboard_util import writer

from torch import nn as nn
import torchvision.models as models

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from random import randrange as r


def show_layer(img, w, h):
    fig = plt.figure(figsize=(w, h))
    gs = gridspec.GridSpec(w, h)
    gs.update(wspace=0.025, hspace=0.025)
    for c in range(img.shape[0]):
        ax = plt.subplot(gs[c])
        plt.axis('off')
        c_img = img[c:c + 1, :, :]
        fig.add_subplot(ax)
        plt.imshow(c_img[0].detach().numpy(), interpolation='nearest')


def visualize(model_name, dataset_folder="dataset", img_size=224, normalize=False):

    _, _, _, test_loader = set_dataset_and_loaders(dataset_folder, batch_size=1,
                                                   img_size=img_size, num_workers=4, normalize=normalize)

    for image, _ in test_loader:

        image = image.to(device)

        plt.imshow(image[0].permute(1, 2, 0).detach().numpy(), interpolation='nearest')
        # writer.add_image(tag="initial", img_tensor=image[0])

        if model_name == models.resnet18.__name__:
            model = models.resnet18()

            image = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)(image)
            show_layer(image[0], 8, 8)
            # conv1_fig.canvas.draw()
            # arr = np.array(conv1_fig.canvas.renderer.buffer_rgba())
            # writer.add_image(tag="conv1", img_tensor=arr)

            image = model.layer1(image)
            show_layer(image[0], 8, 8)

            image = model.layer2(image)
            show_layer(image[0], 8, 16)

            image = model.layer3(image)
            show_layer(image[0], 16, 16)

            image = model.layer4(image)
            show_layer(image[0], 16, 32)

        elif model_name == models.vgg16.__name__:
            model = models.vgg16()

            image = nn.Sequential(*[model.features[i] for i in range(5)])(image)
            show_layer(image[0], 8, 8)

            image = nn.Sequential(*[model.features[i] for i in range(5, 10)])(image)
            show_layer(image[0], 8, 16)

            image = nn.Sequential(*[model.features[i] for i in range(10, 17)])(image)
            show_layer(image[0], 16, 16)

            image = nn.Sequential(*[model.features[i] for i in range(17, 24)])(image)
            show_layer(image[0], 16, 32)

            image = nn.Sequential(*[model.features[i] for i in range(24, 31)])(image)
            show_layer(image[0], 16, 32)

        elif model_name == models.alexnet.__name__:
            model = models.alexnet()

            image = nn.Sequential(*[model.features[i] for i in range(3)])(image)
            show_layer(image[0], 8, 8)

            image = nn.Sequential(*[model.features[i] for i in range(3, 6)])(image)
            show_layer(image[0], 12, 16)

            image = nn.Sequential(*[model.features[i] for i in range(6, 8)])(image)
            show_layer(image[0], 16, 24)

            image = nn.Sequential(*[model.features[i] for i in range(8, 10)])(image)
            show_layer(image[0], 16, 16)

            image = nn.Sequential(*[model.features[i] for i in range(10, 13)])(image)
            show_layer(image[0], 16, 16)

        plt.show()
        break


if __name__ == '__main__':
    visualize("vgg16", "dataset", img_size=224, normalize=True)
