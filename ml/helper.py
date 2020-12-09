import numpy as np
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV

from ml.dataset import read_dataset, divide_dataset

from util.logger_util import log
from util.tensorboard_util import writer


def get_prediction(model, X_ts, y_ts):
    # prediction
    # test_set_size = len(y_ts)
    #
    # false = 0
    # y_predict = list(model.predict([X_ts][0]))
    # for i in range(0, test_set_size - 1):
    #     if y_ts[i] != y_predict[i]:
    #         false += 1
    #
    # success_ratio = ((test_set_size - false) / test_set_size) * 100
    # log.info("Test Success Ratio: " + str(success_ratio) + '%')

    log.info("Test Success Ratio: " + str(100 * model.score(X_ts, y_ts)) + '%')


def get_prediction_cv(seed, model, X, y, cv):
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    get_prediction_kf(kf, model, X, y)


def get_prediction_kf(kf, model, X, y, tag=None):
    cv = kf.n_splits
    ratios = []
    for e, (train, test) in enumerate(kf.split(X, y)):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X_train, X_test, y_train, y_test = X[train], X[test], np.array(y)[train], np.array(y)[test]

        model.fit(X_train, y_train)
        success_ratio = model.score(X_test, y_test)
        log.info(str(cv) + "-Fold CV -- Iteration " + str(e) + " Test Success Ratio: " + str(100*success_ratio) + "%")
        ratios.append(success_ratio)
        if tag is not None:
            writer.add_scalar(tag, success_ratio, e)

    log.info(str(cv) + "-Fold CV Average Test Success Ratio: " + str(100 * np.average(np.array(ratios))) + "%")


def get_dataset(model_name, dataset_folder, img_size, normalize, divide=False):
    log.info("Reading dataset")
    X, y = read_dataset(dataset_folder=dataset_folder, resize_value=(img_size, img_size), to_crop=True)

    if normalize:
        X = StandardScaler().fit_transform(X)

    if divide:
        log.info("Dividing dataset into train and test data")
        X_tr, y_tr, X_ts, y_ts = divide_dataset(X, y)
        log.info("Train data length: %d" % len(y_tr))
        log.info("Test data length: %d" % len(y_ts))

        return X_tr, y_tr, X_ts, y_ts

    return X, y


def get_best_lambda(classifier, grad_dict, cv, X, y):
    pipe = Pipeline([('classifier', classifier)])

    param_grid = [grad_dict]

    clf = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=1, n_jobs=4, pre_dispatch=2*4)

    best_lambda = clf.fit(X, y)

    return best_lambda
