from cnn.dataset import inv_normalize
from cnn.helper import set_dataset_and_loaders
from cnn.model import run_model

from util.garbage_util import collect_garbage
from util.logger_util import log
from util.tensorboard_util import writer


def main(save=False, dataset_folder="dataset", pretrain_file=None, augmented=False, batch_size=20, img_size=227, num_workers=4, model_name='alexnet', optimizer_name='Adam', is_pre_trained=False, fine_tune=False, num_epochs=18, normalize=None, validation_freq=0.1):

    if not is_pre_trained and fine_tune:
        fine_tune = False

    if not is_pre_trained and pretrain_file is not None:
        pretrain_file = None

    log.info("Constructing datasets and loaders")
    train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder, augmented, batch_size, img_size, num_workers, normalize)

    set_0, set_1 = 0, 0
    for imgs, labels in test_loader:
        if set_0 == 3 and set_1 == 3:
            break

        for e, label in enumerate(labels.tolist()):
            if label == 0 and set_0 != 3:
                writer.add_image("{} - class image sample {}".format(train_data.classes[0], set_0), inv_normalize(imgs[e], normalize))
                set_0 += 1
            elif label == 1 and set_1 != 3:
                writer.add_image("{} - class image sample {}".format(train_data.classes[1], set_1), inv_normalize(imgs[e], normalize))
                set_1 += 1

            if set_0 == 3 and set_1 == 3:
                break

    log.info("Calling the model: " + model_name)
    run_model(model_name=model_name, optimizer_name=optimizer_name, is_pre_trained=is_pre_trained, fine_tune=fine_tune, train_loader=train_loader, test_loader=test_loader, num_epochs=num_epochs, save=save, dataset_folder=dataset_folder, pretrain_file=pretrain_file, validation_freq=validation_freq)

    collect_garbage()
    writer.close()


if __name__ == '__main__':
    save = False
    log.info("Process Started")
    main(model_name="alexnet", is_pre_trained=True, save=False, batch_size=64, num_epochs=5, optimizer_name="Adam", img_size=112, validation_freq=0.2)
    log.info("Process Finished")
