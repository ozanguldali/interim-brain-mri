import math
import random

import torch
from torch import optim
from tqdm.notebook import tqdm, trange

from cnn import device
from cnn.util import divide_chunks

from util.logger_util import log
from util.tensorboard_util import writer

from cnn.validate import validate_model


def train_model(model, train_loader, test_loader, metric, optimizer, validation_freq, num_epochs=25, update_loss=False):
    learning_rate = 0.1
    total_loss_history = []
    total_acc_history = []
    validate_every = max(1, math.floor(num_epochs * validation_freq))
    last_validate_iter = 0

    log.info("Training the model")
    # Iterate through train set mini batches
    for epoch in trange(num_epochs):
        correct = 0
        total = len(train_loader.dataset)
        update = update_loss
        loss_history = []
        acc_history = []
        for e, (images, labels) in enumerate(train_loader):  # tqdm
            # zero the parameter gradients
            optimizer.zero_grad()

            inputs = images.to(device)
            labels = labels.to(device)

            # Do the forward pass
            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            truths = torch.sum((predictions == labels).float())
            correct += truths.item()

            loss = metric(outputs, labels)
            loss_history.append(loss.item())

            if update \
                    and (epoch != 0 and epoch != num_epochs - 1)\
                    and e == len(train_loader) - 1 \
                    and (epoch + 1) % int(num_epochs / 4) == 0:
                update = False
                learning_rate = float(learning_rate / 10)
                log.info("learning rate is updated to " + str(learning_rate))
                optimizer = optim.Adam(optimizer.param_groups, lr=learning_rate)

            # Calculate gradients and step
            loss.backward()
            optimizer.step()

        log.info("\nIteration number on epoch %d / %d is %d" % (epoch + 1, num_epochs, len(loss_history)))
        epoch_loss = sum(loss_history) / len(loss_history)
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        total_loss_history.append(epoch_loss)
        epoch_acc = correct / total
        writer.add_scalar("Acc/Train", epoch_acc, epoch)
        total_acc_history.append(epoch_acc)
        log.info("Epoch {} --> training loss: {} - training acc: {}"
                 .format(epoch + 1,
                         round(epoch_loss, 4),
                         round(epoch_acc, 4)))

        if epoch % validate_every == 0 and epoch != (num_epochs-1):
            last_validate_iter = int(epoch / validate_every)
            validate_model(model, test_loader, metric, last_validate_iter)
            model = model.train()
            metric = metric.train()

    log.info("\nTotal training iteration: %d" % len(total_loss_history))
    total_loss = sum(total_loss_history) / len(total_loss_history)
    total_acc = sum(total_acc_history) / len(total_acc_history)
    log.info("Average --> training loss: {} - training acc: {} "
             .format(round(total_loss, 6),
                     round(total_acc, 6)))

    return last_validate_iter


def train_fine_tuned_model(convolutional, classifier, train_loader, metric, optimizer, num_epochs=25, update_loss=False):
    learning_rate = 0.0001
    batch_size = train_loader.batch_size

    total_loss_history = []
    features = []
    classes = []

    log.info("Extracting features by frozen convolution base:")
    for (images, labels) in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        features.extend(convolutional(images))
        classes.extend(labels)
        # clear_cpu()

    train_matrix = list(zip(features, classes))

    log.info("Training the model")
    # Iterate through train set mini batches
    for epoch in trange(num_epochs):
        random.shuffle(train_matrix)
        update = update_loss
        loss_history = []

        chunked_generator = divide_chunks(train_matrix, batch_size)
        chunked = list(chunked_generator)
        batching_len = len(chunked)
        for e, batch in enumerate(chunked):  # tqdm
            images, labels = [k for k, _ in batch], [v for _, v in batch]
            images = torch.stack(images).detach()
            labels = torch.stack(labels).detach()

            optimizer.zero_grad()
            # inputs = images.to(device)
            # labels = labels.to(device)

            # Do the forward pass
            outputs = classifier(images)
            loss = metric(outputs, labels)
            loss_history.append(loss.item())
            total_loss_history.append(loss.item())

            if update \
                    and (epoch != 0 and epoch != num_epochs - 1) \
                    and e == batching_len - 1 \
                    and (epoch + 1) % int(num_epochs / 4) == 0:
                update = False
                learning_rate = float(learning_rate / 10)
                log.info("learning rate is updated to " + str(learning_rate))
                optimizer = optim.Adam(optimizer.param_groups, lr=learning_rate)

            # Calculate gradients and step
            loss.backward()
            optimizer.step()

        # log.info("Iteration number on epoch %d / %d is %d" % (epoch + 1, num_epochs, len(loss_history)))
        # log.info("Average loss on epoch " + str(epoch + 1) + " : " +
        #          str(round(sum(loss_history) * 100 / len(loss_history), 4)) + " %\n")

    log.info("\nTotal training iteration: %d" % len(total_loss_history))
    log.info("Average training loss: " + str(round(sum(total_loss_history) * 100 / len(total_loss_history), 4)) + " %")
