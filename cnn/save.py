import torch

from cnn import ROOT_DIR
from cnn.architect import InterimNet


def save_model(model, path):
    torch.save(model.state_dict(), ROOT_DIR + "/" + path)


if __name__ == '__main__':
    save_model(InterimNet(), "/cnn/dataset_out.pth")
