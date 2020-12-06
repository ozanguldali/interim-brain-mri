import sys

from ml.util import run_svm, run_lr, run_knn

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, X, y, penalty, kf, lambdas, seed):
    collect_garbage()

    if model_name == "svm":
        run_svm(seed, X=X, y=y, penalty=penalty, kf=kf,
                lambdas=(lambdas if (penalty or penalty is None) else None))

    elif model_name == "lr":
        run_lr(seed, X=X, y=y, penalty=penalty, kf=kf,
               lambdas=(lambdas if (penalty or penalty is None) else None))

    elif model_name == "knn":
        if penalty is None or penalty:
            log.info("LASSO Penalty approach is not used on KNN. Thus the run is performing with Penalty: False")

        run_knn(X=X, y=y, penalty=False, kf=kf)

    elif model_name == "all":
        log.info("Running CNN model: svm")
        run_svm(seed, X=X, y=y, penalty=penalty, kf=kf,
                lambdas=(lambdas if (penalty or penalty is None) else None))

        log.info("Running CNN model: lr")
        run_lr(seed, X=X, y=y, penalty=penalty, kf=kf,
               lambdas=(lambdas if (penalty or penalty is None) else None))

        log.info("Running CNN model: knn")
        run_knn(X=X, y=y, penalty=False, kf=kf)

    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)