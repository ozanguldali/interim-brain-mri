import sys

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from cnn.helper import get_grad_update_params
from optim import padam

from cnn import device, ROOT_DIR, SAVE_FILE, MODEL_NAME
import cnn.architect as architect
from cnn.load import load_model
from cnn.save import save_model
from cnn.summary import get_summary, get_fine_tuned_summary
from cnn.test import test_model
from cnn.train import train_model, train_fine_tuned_model
from cnn.util import prepare_alexnet, prepare_resnet, prepare_vgg, prepare_densenet, is_verified
from util.file_util import path_exists

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, optimizer_name, is_pre_trained, fine_tune, train_loader, test_loader, validation_freq, lr,
              momentum, partial, betas, weight_decay, update_lr=True, num_epochs=25, save=False,
              dataset_folder="dataset", pretrain_file=None):
    collect_garbage()

    MODEL_NAME[0] = model_name

    num_classes = len(train_loader.dataset.classes)

    # instantiate the model
    frozen, evaluation = None, None
    log.info("Instantiate the model")
    if model_name == architect.procnn.__name__:
        if is_pre_trained and pretrain_file is None:
            log.fatal("Pretrain File must be defined on manual cnn architects.")
            sys.exit(1)

        model = architect.procnn(pretrained=is_pre_trained, dataset_folder=pretrain_file)
        if is_pre_trained:
            frozen = model.features
            evaluation = nn.Sequential(model.flatten, model.fc1, model.fc2, model.fc3, model.softMax)

    elif model_name == models.alexnet.__name__:
        model, frozen, evaluation = prepare_alexnet(is_pre_trained, pretrain_file, fine_tune, num_classes)

    elif model_name in (models.resnet18.__name__, models.resnet50.__name__, models.resnet152.__name__):
        model, frozen, evaluation = prepare_resnet(model_name, is_pre_trained, pretrain_file, fine_tune, num_classes)

    elif model_name in (models.vgg16.__name__, models.vgg19.__name__):
        model, frozen, evaluation = prepare_vgg(model_name, is_pre_trained, pretrain_file, fine_tune, num_classes)

    elif model_name == models.densenet169.__name__:
        model, frozen, evaluation = prepare_densenet(is_pre_trained, pretrain_file, fine_tune, num_classes)

    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    log.info("Setting the model to device")
    # if is_pre_trained and fine_tune:
    #     frozen = frozen.to(device)
    #     evaluation = evaluation.to(device)
    #     model = nn.Sequential(frozen, evaluation)
    # else:
    #     model = model.to(device)

    model = model.to(device)

    if "densenet" not in model_name:
        log.info("The summary:")
        # # model = reduce_features(model)
        # if is_pre_trained and fine_tune:
        #     get_fine_tuned_summary(frozen, evaluation, train_loader)
        # else:
        #     get_summary(model, train_loader)
        get_summary(model, train_loader)

    collect_garbage()

    log.info("Setting the metric")
    metric = nn.CrossEntropyLoss()

    model_parameters = get_grad_update_params(model, fine_tune)

    if optimizer_name == optim.Adam.__name__:
        optimizer = optim.Adam(model_parameters, lr=lr)
    elif optimizer_name == optim.SGD.__name__:
        optimizer = optim.SGD(model_parameters, lr=lr, momentum=momentum)
    elif optimizer_name == optim.AdamW.__name__:
        optimizer = optim.AdamW(model_parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == optim.AdamW.__name__:
        optimizer = optim.AdamW(model_parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == padam.Padam.__name__:
        optimizer = padam.Padam(model_parameters, lr=lr, partial=partial, weight_decay=weight_decay, betas=betas)
    else:
        log.fatal("not implemented optimizer name: {}".format(optimizer_name))
        sys.exit(1)

    log.info("Setting the optimizer as: {}".format(optimizer_name))

    SAVE_FILE[0] = ("" if not is_pre_trained else "PreTrained_") + model_name + "_" + optimizer_name + "_" + dataset_folder + "_out.pth"

    # last_val_iterator = 0
    # if is_pre_trained and fine_tune:
    #     # frozen = frozen.cpu()
    #     frozen = frozen.eval()
    #     train_fine_tuned_model(frozen, evaluation, train_loader, metric, optimizer, num_epochs=num_epochs,
    #                            update_lr=(model_name == architect.procnn.__name__))
    # else:
    #     # update_lr=(model_name == architect.procnn.__name__)
    #     last_val_iterator = train_model(model, train_loader, test_loader, metric, optimizer, lr=lr,
    #                                     num_epochs=num_epochs, update_lr=update_lr, validation_freq=validation_freq, save=save)
    last_val_iterator = train_model(model, train_loader, test_loader, metric, optimizer, lr=lr,
                                    num_epochs=num_epochs, update_lr=update_lr, validation_freq=validation_freq,
                                    save=save)

    log.info("Testing the model")
    test_acc = test_model(model, test_loader, last_val_iterator)

    if save and is_verified(test_acc):
        exist_files = path_exists(ROOT_DIR, SAVE_FILE[0], "contains")

        better = len(exist_files) == 0
        if not better:
            exist_acc = []
            for file in exist_files:
                exist_acc.append(float(file.split("_")[0].replace(",", ".")))
            better = all(test_acc > acc for acc in exist_acc)
        if better:
            save_model(model=model, path=str(round(test_acc, 2)) + "_" + SAVE_FILE[0])

    return model


def weighted_model(model_name, is_pre_trained, pretrain_file):
    out_file = ROOT_DIR + "/" + pretrain_file + ".pth"

    if model_name == architect.procnn.__name__:
        model = architect.procnn(pretrained=is_pre_trained, dataset_folder=pretrain_file)

    elif model_name == models.alexnet.__name__:
        model = models.alexnet()

    elif model_name == models.resnet18.__name__:
        model = models.resnet18()

    elif model_name == models.resnet50.__name__:
        model = models.resnet50()

    elif model_name == models.resnet152.__name__:
        model = models.resnet152()

    elif model_name == models.vgg16.__name__:
        model = models.vgg16()

    elif model_name == models.vgg19.__name__:
        model = models.vgg19()

    elif model_name == models.densenet169.__name__:
        model = models.densenet169()

    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    return load_model(model, out_file)
