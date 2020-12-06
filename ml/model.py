import sys

from ml.util import run_svm, run_lr, run_knn

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, X_tr, y_tr, X_ts, y_ts, cv, penalty, num_workers):
    collect_garbage()

    if model_name == "svm":
        run_svm(X_tr=X_tr, y_tr=y_tr, X_ts=X_ts, y_ts=y_ts, cv=cv, penalty=penalty)
    elif model_name == "lr":
        run_lr(X_tr=X_tr, y_tr=y_tr, X_ts=X_ts, y_ts=y_ts, cv=cv, penalty=penalty)
    elif model_name == "knn":
        run_knn(X_tr=X_tr, y_tr=y_tr, X_ts=X_ts, y_ts=y_ts, cv=cv, penalty=penalty)
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)
