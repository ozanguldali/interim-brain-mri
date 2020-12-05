import torch
from tqdm.notebook import tqdm

from cnn import device

from util.logger_util import log


def validate_model(model, test_loader, metric, iterator):
    correct = 0
    total = len(test_loader)

    # set the model into evaluation mode
    model = model.eval()

    # behavior of the batch norm layer so that it is not sensitive to batch size
    with torch.set_grad_enabled(False):
        # Iterate through test set mini batches
        for e, (images, labels) in enumerate(tqdm(test_loader)):
            # Forward pass
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            predictions = torch.argmax(outputs, dim=1)
            correct += torch.sum((predictions == labels).float())
            accuracy = correct / total

            loss = metric(outputs, labels)

    log.info("{}th validation loss and accuracy on epoch: {} - {} "
             .format(iterator,
                     round(loss.item(), 4),
                     round(accuracy.item(), 4)))

    return 100 * accuracy
