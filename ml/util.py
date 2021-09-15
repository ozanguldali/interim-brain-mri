from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ml.helper import get_prediction_kf

from util.logger_util import log


def run_svm(X, y, kf=None):
    tag = "Acc/SVM"

    svc_cv = SVC(probability=True)

    get_prediction_kf(kf, svc_cv, X, y, tag)
    log.info("")


def run_lr(X, y, kf=None):
    tag = "Acc/LR"

    clf_cv = LogisticRegression(max_iter=100000)

    get_prediction_kf(kf, clf_cv, X, y, tag)
    log.info("")


def run_knn(X, y, kf=None):
    tag = "Acc/KNN"

    neigh_cv = KNeighborsClassifier(n_neighbors=len(set(y)))

    get_prediction_kf(kf, neigh_cv, X, y, tag)
    log.info("")
