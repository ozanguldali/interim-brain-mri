from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MyData(Dataset):
    transform = transforms.Compose(
        [
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    def __init__(self, data, target, transform=transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if (y == 0) and self.transform:  # check for minority class
            x = self.transform(x)

        return x, y
