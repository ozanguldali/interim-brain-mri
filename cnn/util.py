import sys

from torch import nn
from torchvision import models

from cnn import ROOT_DIR
from cnn.load import load_model

from util.logger_util import log


def divide_chunks(target_list, n):
    # looping till length target_list
    for i in range(0, len(target_list), n):
        yield target_list[i:i + n]


def prepare_alexnet(is_pre_trained, pretrain_file, fine_tune, num_classes):
    frozen, evaluation = None, None

    if pretrain_file is None:
        model = models.alexnet(pretrained=is_pre_trained,
                               num_classes=1000 if is_pre_trained else num_classes)
        if fine_tune:
            frozen = nn.Sequential(
                *[model.features[i] for i in range(3)]
            )
            evaluation = nn.Sequential(
                *[model.features[i] for i in range(3, len(model.features))],
                model.avgpool,
                nn.Flatten(),
                model.classifier
            )

    else:  # 'pretrain_file is None' statement equals to 'is_pre_trained == True'
        model = models.alexnet(num_classes=1000 if "PreTrained" in pretrain_file else num_classes)
        model = load_model(model, ROOT_DIR + "/" + pretrain_file + ".pth")

        if fine_tune:
            frozen = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten()
            )
            evaluation = model.classifier

    return model, frozen, evaluation


def prepare_resnet(model_name, is_pre_trained, pretrain_file, fine_tune, num_classes):
    frozen, evaluation = None, None

    if pretrain_file is None:
        if model_name == models.resnet18.__name__:
            model = models.resnet18(pretrained=is_pre_trained,
                                    num_classes=1000 if is_pre_trained else num_classes)
        elif model_name == models.resnet50.__name__:
            model = models.resnet50(pretrained=is_pre_trained,
                                    num_classes=1000 if is_pre_trained else num_classes)
        elif model_name == models.resnet152.__name__:
            model = models.resnet152(pretrained=is_pre_trained,
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
            evaluation = nn.Sequential(
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool,
                nn.Flatten(),
                model.fc
            )

    else:  # 'pretrain_file is None' statement equals to 'is_pre_trained == True'
        if model_name == models.resnet18.__name__:
            model = models.resnet18(num_classes=1000)
        elif model_name == models.resnet50.__name__:
            model = models.resnet50(num_classes=1000)
        elif model_name == models.resnet152.__name__:
            model = models.resnet152(num_classes=1000)
        else:
            log.fatal("model name is not known: " + model_name)
            sys.exit(1)

        model = load_model(model, ROOT_DIR + "/" + pretrain_file + ".pth")

        if fine_tune:
            frozen = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool,
                nn.Flatten()
            )
            evaluation = nn.Sequential(
                model.fc
            )

    return model, frozen, evaluation


def prepare_vgg(model_name, is_pre_trained, pretrain_file, fine_tune, num_classes):
    frozen, evaluation = None, None

    if pretrain_file is None:
        if model_name == models.vgg16.__name__:
            model = models.vgg16(pretrained=is_pre_trained,
                                 num_classes=1000 if is_pre_trained else num_classes)
        elif model_name == models.vgg19.__name__:
            model = models.vgg19(pretrained=is_pre_trained,
                                 num_classes=1000 if is_pre_trained else num_classes)
        else:
            log.fatal("model name is not known: " + model_name)
            sys.exit(1)

        if fine_tune:
            frozen = nn.Sequential(
                *[model.features[i] for i in range(5)]
            )
            evaluation = nn.Sequential(
                *[model.features[i] for i in range(5, len(model.features))],
                model.avgpool,
                nn.Flatten(),
                model.classifier
            )

    else:  # 'pretrain_file is None' statement equals to 'is_pre_trained == True'
        if model_name == models.vgg16.__name__:
            model = models.vgg16(num_classes=1000)
        elif model_name == models.vgg19.__name__:
            model = models.vgg19(num_classes=1000)
        else:
            log.fatal("model name is not known: " + model_name)
            sys.exit(1)

        model = load_model(model, ROOT_DIR + "/" + pretrain_file + ".pth")

        if fine_tune:
            frozen = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten(),
            )
            evaluation = nn.Sequential(
                model.classifier
            )

    return model, frozen, evaluation


def prepare_densenet(is_pre_trained, pretrain_file, fine_tune, num_classes):
    frozen, evaluation = None, None

    if pretrain_file is None:
        model = models.densenet169(pretrained=is_pre_trained,
                                   num_classes=1000 if is_pre_trained else num_classes)

        if fine_tune:
            frozen = nn.Sequential(
                *[model.features[i] for i in range(4)]
            )
            evaluation = nn.Sequential(
                *[model.features[i] for i in range(4, len(model.features))],
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                model.classifier
            )

    else:  # 'pretrain_file is None' statement equals to 'is_pre_trained == True'
        model = models.densenet169(num_classes=1000)
        model = load_model(model, ROOT_DIR + "/" + pretrain_file + ".pth")

        if fine_tune:
            frozen = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten()
            )
            evaluation = model.classifier

    return model, frozen, evaluation


def is_verified(model, acc):
    model_name = model.__name__

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
