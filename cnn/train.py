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
    learning_rate = 0.0001
    total_loss_history = []
    total_accuracy_history = []
    validate_every = max(1, math.floor(num_epochs * validation_freq))

    log.info("Training the model")
    # Iterate through train set mini batches
    for epoch in trange(num_epochs):
        update = update_loss
        loss_history = []
        accuracy_history = []
        correct = 0
        total = len(train_loader)

        for e, (images, labels) in enumerate(train_loader):  # tqdm
            # zero the parameter gradients
            optimizer.zero_grad()

            inputs = images.to(device)
            labels = labels.to(device)
            writer.add_image("{} - input image".format(labels[0]), inputs[0], epoch)

            # Do the forward pass
            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            correct += torch.sum((predictions == labels).float())
            accuracy = correct / total
            accuracy_history.append(accuracy.item())
            total_accuracy_history.append(accuracy.item())
            writer.add_scalar("Accuracy/train", accuracy, epoch)

            loss = metric(outputs, labels)
            loss_history.append(loss.item())
            total_loss_history.append(loss.item())
            writer.add_scalar("Loss/train", loss, epoch)

            if update \
                    and (epoch != 0 and epoch != num_epochs - 1) \
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
        log.info("Average training loss and accuracy on epoch {}: {} - {} "
                 .format(epoch + 1,
                         round(sum(loss_history) / len(loss_history), 4),
                         round(sum(accuracy_history) / len(accuracy_history), 4)))

        if epoch % validate_every == 0 and epoch != (num_epochs - 1):
            validate_model(model, test_loader, metric, int(epoch / validate_every))
            model = model.train()
            torch.set_grad_enabled(True)

    log.info("\nTotal training iteration: %d" % len(total_loss_history))
    log.info("Average training loss: {} - {} "
             .format(round(sum(total_loss_history) / len(total_loss_history), 4),
                     round(sum(total_accuracy_history) / len(total_accuracy_history), 4)))

    # writer.flush()


def train_fine_tuned_model(convolutional, classifier, train_loader, metric, optimizer, num_epochs=25,
                           update_loss=False):
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
