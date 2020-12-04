import torch
from tqdm.notebook import tqdm

from cnn import device

from util.logger_util import log


def validate_model(model, test_loader, iterator):
    correct = 0
    total = len(test_loader)

    # set the model into evaluation mode
    model = model.eval()

    # behavior of the batch norm layer so that it is not sensitive to batch size
    with torch.no_grad():
        # Iterate through test set mini batches
        for e, (images, labels) in enumerate(tqdm(test_loader)):
            # Forward pass
            inputs = images.to(device)
            labels = labels.to(device)
            y = model(inputs)

            predictions = torch.argmax(y, dim=1)
            correct += torch.sum((predictions == labels).float())

    log.info('\n{}th Validation accuracy: {}'.format(iterator, correct / total))

    return 100 * correct / total