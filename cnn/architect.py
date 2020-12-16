import torch
import torch.nn as nn

from cnn import device

__all__ = ['InterimNet', 'interimnet']


class InterimNet(nn.Module):
    def __init__(self):
        super(InterimNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, stride=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=128),
            nn.AdaptiveMaxPool2d(output_size=37),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),
            nn.AdaptiveMaxPool2d(output_size=10),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),
            nn.AdaptiveMaxPool2d(output_size=3),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),
            nn.AdaptiveMaxPool2d(output_size=1),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),
            nn.AdaptiveMaxPool2d(output_size=1),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(1 * 1 * 512, 1024)
        )

        self.fc2 = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(1 * 1 * 1024, 1024)
        )

        self.fc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(1 * 1 * 1024, 2)
        )

        self.softMax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softMax(x)

        return x


def interimnet(pretrained=False, dataset_file="dataset", **kwargs):
    r"""ProCNN model architecture from the

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param dataset_file: stored weights file
    """
    model = InterimNet()

    if pretrained:
        map_location = None if torch.cuda.is_available() else device
        model.load_state_dict(torch.load(dataset_file, map_location=map_location))

    return model
