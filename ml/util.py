import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from rpy2.robjects import r

import cnn
from cnn.features import feature_clean
from ml import ROOT_DIR
from ml.helper import get_prediction, get_prediction_cv, get_best_lambda, get_prediction_kf

from util.logger_util import log


def run_svm(seed, X, y, penalty, lambdas=None, kf=None):
    cv = kf.n_splits

    if penalty is None:
        run_svm(seed, X=X, y=y, penalty=False, kf=kf)
        run_svm(seed, X=X, y=y, penalty=True, kf=kf, lambdas=lambdas)
    else:
        tag = "Acc/SVM" + ("/LASSO" if penalty else "")
        log.info("Penalty Enabled: " + str(penalty))

        if penalty:
            grad_dict = {
                'classifier': [LinearSVC()],
                'classifier__penalty': ['l1'],
                'classifier__C': lambdas,
                'classifier__dual': [False],
                'classifier__random_state': [seed],
                'classifier__max_iter': [10000]
            }
            bests = get_best_lambda(LinearSVC(), grad_dict, cv, X, y)
            best_lambda = lambdas[bests.best_index_]
            log.info("Best lambda value has determined as: " + str(best_lambda))
            svc_cv = LinearSVC(max_iter=100000, penalty='l1', dual=False, C=best_lambda)
        else:
            svc_cv = SVC()

        get_prediction_kf(kf, svc_cv, X, y, tag)
        log.info("")


def run_lr(seed, X, y, penalty, lambdas=None, kf=None):
    cv = kf.n_splits

    if penalty is None:
        run_lr(seed, X=X, y=y, penalty=False, kf=kf)
        run_lr(seed, X=X, y=y, penalty=True, kf=kf, lambdas=lambdas)
    else:
        tag = "Acc/LR" + ("/LASSO" if penalty else "")
        log.info("Penalty Enabled: " + str(penalty))

        if penalty:
            grad_dict = {
                'classifier': [LogisticRegression()],
                'classifier__penalty': ['l1'],
                'classifier__C': lambdas,
                'classifier__solver': ["liblinear"],
                'classifier__random_state': [seed],
                'classifier__max_iter': [10000]
            }
            bests = get_best_lambda(LogisticRegression(), grad_dict, cv, X, y)
            best_lambda = lambdas[bests.best_index_]
            log.info("Best lambda value has determined as: " + str(best_lambda))
            clf_cv = LogisticRegression(max_iter=100000, solver='liblinear', penalty='l1', C=best_lambda)
        else:
            clf_cv = LogisticRegression(max_iter=100000, solver='liblinear')

        get_prediction_kf(kf, clf_cv, X, y, tag)
        log.info("")


def run_knn(X, y, kf, penalty=False):
    if penalty is None:
        run_knn(X, y, kf)
        run_knn(X, y, kf)
    else:
        tag = "Acc/KNN" + ("/LASSO" if penalty else "")
        log.info("Penalty Enabled: " + str(penalty))

        neigh_cv = KNeighborsClassifier(n_neighbors=len(set(y)), p=(1 if penalty else 2))

        get_prediction_kf(kf, neigh_cv, X, y, tag)
        log.info("")
