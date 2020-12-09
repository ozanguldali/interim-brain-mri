import torch

from cnn import ROOT_DIR
from cnn.architect import ProCNN


def save_model(model, path):
    torch.save(model.state_dict(), ROOT_DIR + "/" + path)


if __name__ == '__main__':
    save_model(ProCNN(), "/cnn/dataset_out.pth")
