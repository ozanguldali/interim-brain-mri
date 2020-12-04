import numpy as np

from torch import nn
from tqdm import tqdm

from cnn import device
from util.logger_util import log


def reduce_features(model):
    # Need to know the number of features
    feature_size = model.fc.in_features
    # coming out of the penultimate layer and into the fully connected one.
    model.fc = nn.Linear(feature_size, 2)

    return model


def extract_features(data_loader, feature_extractor):
    X, y = [], []

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        X.extend(
            [features.tolist() for features in feature_extractor(images)]
        )
        y.extend(labels.tolist())

    return X, y


def alexnet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *[model.classifier[i] for i in range(3)]  # ,
        # nn.Linear(4096, 256),
        # nn.ReLU(inplace=True)
    )

    return feature_extractor


def resnet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        nn.Flatten(),
    )

    return feature_extractor


def vgg_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *[model.classifier[i] for i in range(3)]
    )

    return feature_extractor


def densenet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten()
    )

    return feature_extractor


def feature_clean(X_cnn, y, class0_size, class1_size, class0):
    np_x_cnn = np.reshape(X_cnn, (len(X_cnn), len(X_cnn[0])))

    nr, nc = np_x_cnn.shape

    zero_std = []
    for c in range(nc):
        column_i_class0 = np.zeros((1, class0_size))
        column_i_class1 = np.zeros((1, class1_size))
        index_i_0 = 0
        index_i_1 = 0
        for r in range(nr):
            if y[r] == class0:
                column_i_class0[0, index_i_0] = np_x_cnn[r, c]
                index_i_0 += 1
            else:
                column_i_class1[0, index_i_1] = np_x_cnn[r, c]
                index_i_1 += 1

        if np.max(column_i_class0) == 0.0 or np.max(column_i_class1) == 0.0:
            zero_std.append(c)

    len_zero_std = len(zero_std)
    log.info("Number of features having within-class standard deviation as 0: " + str(len_zero_std))
    if len_zero_std != 0:
        log.info("Eliminating 0 within-class standard deviation feature columns")
        np_x_cnn = np.delete(np_x_cnn, zero_std, axis=1)
        del X_cnn
        X_cnn = np.ndarray.tolist(np_x_cnn)

    return X_cnn
