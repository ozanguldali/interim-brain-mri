import os
import sys

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer

import run_CNN
import run_ML
from cnn import model as cnn_model, device
from cnn.dataset import set_loader
from cnn.features import extract_features, feature_clean
from cnn.helper import set_dataset_and_loaders, get_feature_extractor
from ml.model import run_model
from ml.util import run_svm, run_lr, run_knn
from util.garbage_util import collect_garbage
from util.logger_util import log

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))


def main(transfer_learning, method="", ml_model_name="", cv=5, penalty: object = False,
         dataset_folder="dataset", pretrain_file=None, batch_size=8, img_size=227, num_workers=4,
         cnn_model_name="", optimizer_name='Adam', validation_freq=0.1, lr=0.001, momentum=0.9, partial=0.125,
         betas=(0.9, 0.99), weight_decay=0.025, is_pre_trained=False, fine_tune=False, num_epochs=16, normalize=True,
         lambdas=None, seed=1):
    if lambdas is None:
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    if not transfer_learning:
        if method.lower() == "ml":
            run_ML.main(ml_model_name, dataset_folder, seed, lambdas, cv, penalty, img_size, normalize)
        elif method.lower() == "cnn":
            run_CNN.main(False, dataset_folder, pretrain_file, False, batch_size, img_size, num_workers,
                         cnn_model_name, is_pre_trained, fine_tune, num_epochs, normalize)
        else:
            log.fatal("method name is not known: " + method)
            sys.exit(1)

    else:
        log.info("Constructing datasets and loaders")
        train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder=dataset_folder,
                                                                                   augmented=False,
                                                                                   batch_size=batch_size,
                                                                                   img_size=img_size,
                                                                                   num_workers=num_workers,
                                                                                   normalize=normalize)

        if is_pre_trained and pretrain_file is not None and\
                dataset_folder in pretrain_file.lower() and \
                cnn_model_name in pretrain_file.lower():
            log.info("Getting PreTrained CNN model: " + cnn_model_name + " from the Weights of " + pretrain_file)
            model = cnn_model.weighted_model(cnn_model_name, is_pre_trained, pretrain_file)

        else:
            log.info("Running CNN model: " + cnn_model_name)
            model = cnn_model.run_model(cnn_model_name, optimizer_name, is_pre_trained, fine_tune, train_loader, test_loader, validation_freq,
                                        lr, momentum, partial, betas, weight_decay, num_epochs, False, dataset_folder, pretrain_file)

        log.info("Feature extractor is creating")
        feature_extractor = get_feature_extractor(cnn_model_name, model.eval())
        log.info("Feature extractor is setting to device: " + str(device))
        feature_extractor = feature_extractor.to(device)

        log.info("Merging CNN train&test datasets")
        dataset = train_data + test_data
        dataset_size = len(dataset)

        class0 = 0
        class1 = 1

        log.info("Constructing loader for merged dataset")
        data_loader = set_loader(dataset=dataset, batch_size=int(len(dataset) / 5), shuffle=False,
                                 num_workers=num_workers)
        log.info("Extracting features as X_cnn array and labels as general y vector")
        X_cnn, y = extract_features(data_loader, feature_extractor)
        class_dist = {i: y.count(i) for i in y}
        class0_size = class_dist[class0]
        class1_size = class_dist[class1]
        log.info("Total class 0 size: " + str(class0_size))
        log.info("Total class 1 size: " + str(class1_size))

        if normalize:
            X_cnn = Normalizer().fit_transform(X_cnn)
        X_cnn = StandardScaler().fit_transform(X_cnn)

        X_cnn = feature_clean(X_cnn, y, class0_size, class1_size, class0)

        log.info("Number of features in X_cnn: " + str(len(X_cnn[0])))

        log.info("Deleting unnecessary variables")
        del model, train_data, train_loader, test_data, test_loader, dataset, data_loader, feature_extractor

        log.info("Creating merged and divided general X_train and X_test arrays")
        X = []
        for c in range(dataset_size):
            row = []
            row.extend(X_cnn[c])
            X.append(row)

        log.info("Number of features in merged X: " + str(len(X[0])))

        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

        run_model(ml_model_name, X, y, penalty, kf, lambdas, seed)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    # main(transfer_learning=True, ml_model_name="svm", penalty=False, cnn_model_name="alexnet", is_pre_trained=True,
    #      dataset_folder="dataset", pretrain_file=None, batch_size=32, num_epochs=1, cv=5, lambdas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    #      seed=23)

    main(transfer_learning=False, method="ml", ml_model_name="all", penalty=None, dataset_folder="dataset",  cv=5,
         lambdas=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0], seed=23)
    log.info("Process Finished")
