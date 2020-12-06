from ml.helper import get_dataset
from ml.model import run_model

from util.garbage_util import collect_garbage
from util.logger_util import log


def main(model_name, dataset_folder, cv=5, penalty=False, img_size=227, normalize=True, num_workers=4):

    log.info("Constructing datasets and arrays")
    X_tr, y_tr, X_ts, y_ts = get_dataset(model_name, dataset_folder, img_size, normalize)

    log.info("Calling the model: " + model_name)
    run_model(model_name, X_tr, y_tr, X_ts, y_ts, cv, penalty, num_workers)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    main(model_name='svm', cv=5, penalty=False, dataset_folder="dataset_notunique", img_size=227, normalize=True)
    log.info("Process Finished")
