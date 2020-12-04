import torch

from cnn import ROOT_DIR
from cnn.architect import ProCNN


def save_model(model, path, optimizer=None):
    # # Initialize optimizer
    # if optimizer is None:
    #     optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #
    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #
    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    torch.save(model.state_dict(), ROOT_DIR + "/" + path)


if __name__ == '__main__':
    save_model(ProCNN(), "/cnn/dataset_out.pth")
