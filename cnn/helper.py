import sys

from torchvision import models

from cnn import ROOT_DIR
from cnn.dataset import set_dataset, set_loader
from cnn.features import alexnet_feature_extractor, resnet_feature_extractor, vgg_feature_extractor, \
    densenet_feature_extractor

from util.logger_util import log


def set_dataset_and_loaders(dataset_folder, augmented, batch_size, img_size, num_workers, normalize=None):

    dataset_dir = ROOT_DIR.split("cnn")[0]

    log.info("Setting train data")
    train_data = set_dataset(folder=dataset_dir + dataset_folder + '/train', size=img_size, augmented=augmented, normalize=normalize)
    log.info("Train data length: %d" % len(train_data))
    log.info("Setting test data")
    test_data = set_dataset(folder=dataset_dir + dataset_folder + '/test', size=img_size, normalize=normalize)
    log.info("Test data length: %d" % len(test_data))

    log.info("Setting train loader")
    train_loader = set_loader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    log.info("Setting test loader")
    test_loader = set_loader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, train_loader, test_data, test_loader


def get_feature_extractor(model_name, model):

    if model_name == models.alexnet.__name__:
        feature_extractor = alexnet_feature_extractor(model)

    elif model_name in (models.resnet18.__name__, models.resnet50.__name__, models.resnet152.__name__):
        feature_extractor = resnet_feature_extractor(model)

    elif model_name in (models.vgg16.__name__, models.vgg19.__name__):
        feature_extractor = vgg_feature_extractor(model)

    elif model_name == models.densenet169.__name__:
        feature_extractor = densenet_feature_extractor(model)

    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    return feature_extractor
