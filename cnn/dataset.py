import csv

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets, transforms

from cnn import ROOT_DIR


def set_dataset(folder, size=224, augmented=False, normalize=None):

    augmenting_list = [
        transforms.RandomHorizontalFlip()
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation((90, 90)),
        # transforms.RandomRotation((180, 180)),
        # transforms.RandomRotation((270, 270))
    ]

    transform = set_transform(resize=size, crop=size, normalize=normalize)
    dataset = datasets.ImageFolder(folder, transform=transform)

    if augmented:
        for augment_type in augmenting_list:
            transform = set_transform(resize=size, crop=size, additional=[augment_type])
            dataset_augmented = datasets.ImageFolder(folder, transform=transform)

            # dataset_unique = tuple(
            #     set(
            #         dataset_unique
            #     ).
            #     union(set(
            #         dataset_augmented
            #         )
            #     )
            # )

            dataset += dataset_augmented
            del dataset_augmented

    return dataset


def set_age_sex(dataset):
    root_dir = ROOT_DIR.split("cnn")[0]
    metadata_path = root_dir + "metadata.csv"

    image_paths = []
    image_paths.extend([img_tuple[0].split("/")[-1] for img_tuple in dataset.datasets[0].imgs])
    image_paths.extend([img_tuple[0].split("/")[-1] for img_tuple in dataset.datasets[1].imgs])

    age_list = [0] * len(image_paths)
    sex_list = [0] * len(image_paths)

    with open(metadata_path, mode='r') as csv_data:
        reader = csv.DictReader(csv_data, delimiter=',')

        for row in reader:
            if row["sex"] != "" and row["age"] != "" and row["filename"] in image_paths:
                index = image_paths.index(row["filename"])
                age_list[index] = int(row["age"])
                sex_list[index] = 1 if row["sex"] == "F" else 0

    return age_list, sex_list


def set_loader(dataset, batch_size=1, shuffle=False, num_workers=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def set_transform(resize=224, crop=224, normalize=None, additional=None):
    if normalize is None or normalize is True:
        normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    transform_list = [
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.Grayscale(3)
    ]

    if additional is not None:
        transform_list.extend(additional)

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize[0], std=normalize[1])
    ])

    return transforms.Compose(transform_list)
