import torch
from tqdm.notebook import tqdm

from sklearn.metrics import confusion_matrix
from cnn import device

from util.logger_util import log
from util.tensorboard_util import writer


def test_model(model, test_loader, iterator=0):
    correct = 0
    total = len(test_loader.dataset)

    prediction_list, label_list = [], []

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
            prediction_list.extend(predictions)
            label_list.extend(labels)

            truths = torch.sum((predictions == labels).float()).item()
            correct += truths

    log.info("Confusion Matrix:")
    log.info(confusion_matrix([p.detach() for p in prediction_list], [l.detach() for l in label_list]))

    acc = (correct / total)
    log.info('\nTest accuracy: {}'.format(acc))
    if iterator != 0:
        writer.add_scalar("Acc/Validation", acc, iterator)

    return 100 * acc

    # model.eval()
    # total = len(test_data)
    # labels = []
    # results_list = []
    # generated = []
    # correct = 0
    # feature_extractor = nn.Sequential(model.features, model.flatten, model.fc1, model.fc2, model.fc3, model.softMax)
    # with torch.no_grad():
    #     for _, data in enumerate(test_data):
    #         actual = data[1]
    #         labels.append(actual)
    #         results = feature_extractor(data[0].reshape(-1, data[0].size(0), data[0].size(1), data[0].size(2)))
    #         results_list.append(results.tolist())
    #         ratio_0 = results[0].tolist()[0]
    #         ratio_1 = results[0].tolist()[1]
    #         if ratio_0 < ratio_1:
    #             expected = 0
    #         else:
    #             expected = 1
    #
    #         generated.append(expected)
    #
    #         if actual == expected:
    #             correct += 1
    # print(labels)
    # print(results_list)
    # print(generated)
    #
    # print('\nTest accuracy soft max: {}'.format(correct / total))