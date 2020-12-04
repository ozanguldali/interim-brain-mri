import os
import imageio
import matplotlib.pyplot as plt

from cnn.dataset import set_transform, set_dataset


def plot(transformed=False):
    if transformed:
        plot_transformed_image()
    else:
        plot_pure_image()


def plot_pure_image():
    cov_ims = os.listdir('./dataset_unique/train/COVID-19/')
    cov_im = imageio.imread('./dataset_unique/train/COVID-19/' + cov_ims[0])
    print(cov_im.shape)

    non_cov_ims = os.listdir('./dataset_unique/train/non-COVID-19/')
    non_cov_im = imageio.imread('./dataset_unique/train/non-COVID-19/' + non_cov_ims[0])
    print(non_cov_im.shape)

    plt.figure()
    plt.imshow(cov_im)
    plt.show()

    plt.figure()
    plt.imshow(non_cov_im)
    plt.show()


def plot_transformed_image():

    train_data = set_dataset(folder='./dataset_unique/train')
    plt.figure()
    plt.imshow(train_data[0][0].permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    plot()
