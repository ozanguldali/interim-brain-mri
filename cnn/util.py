import sys

from torch import nn
from torchvision import models

from cnn import ROOT_DIR, MODEL_NAME
from cnn.helper import set_parameter_requires_grad
from cnn.load import load_model

from util.logger_util import log


def divide_chunks(target_list, n):
    # looping till length target_list
    for i in range(0, len(target_list), n):
        yield target_list[i:i + n]


def prepare_alexnet(is_pre_trained, fine_tune, num_classes):
    model = models.alexnet(pretrained=is_pre_trained,
                           num_classes=1000 if is_pre_trained else num_classes)
    if fine_tune:
        frozen = nn.Sequential(
            *[model.features[i] for i in range(3)]
        )
        set_parameter_requires_grad(frozen)

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


def prepare_resnet(model_name, is_pre_trained, pretrain_file, fine_tune, num_classes):

    if model_name == models.resnet18.__name__:
        model = models.resnet18(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)
    elif model_name == models.resnet50.__name__:
        model = models.resnet50(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    if fine_tune:
        frozen = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        set_parameter_requires_grad(frozen)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def prepare_vgg(model_name, is_pre_trained, fine_tune, num_classes):

    if model_name == models.vgg16.__name__:
        model = models.vgg16(pretrained=is_pre_trained,
                             num_classes=1000 if is_pre_trained else num_classes)
        limit_frozen = 5
    elif model_name == models.vgg19.__name__:
        model = models.vgg19(pretrained=is_pre_trained,
                             num_classes=1000 if is_pre_trained else num_classes)
        limit_frozen = 7
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    if fine_tune:
        frozen = nn.Sequential(
            *[model.features[i] for i in range(limit_frozen)]
        )
        set_parameter_requires_grad(frozen)

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


def prepare_densenet(is_pre_trained, fine_tune, num_classes):

    model = models.densenet169(pretrained=is_pre_trained,
                               num_classes=1000 if is_pre_trained else num_classes)

    if fine_tune:
        frozen = nn.Sequential(
            *[model.features[i] for i in range(4)]
        )
        set_parameter_requires_grad(frozen)

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


def is_verified(acc):
    model_name = MODEL_NAME[0]

    verified = False

    if model_name == models.alexnet.__name__ and acc >= 78:
        verified = True

    elif model_name in (models.resnet18.__name__, models.resnet50.__name__, models.resnet152.__name__) and acc >= 80:
        verified = True

    elif model_name == models.vgg16.__name__ and acc >= 80:
        verified = True

    elif model_name == models.vgg19.__name__ and acc >= 80:
        verified = True

    elif model_name == models.densenet169.__name__ and acc >= 80:
        verified = True

    return verified
