import logging

import numpy as np

from utils.data_loader import DataLoader
from utils.metrics import sum_of_squared_error
from utils.misc import compute_correlation_matrix
from learners.linear_regression import LinearRegression
from utils.k_fold_cross_validation import KFoldCrossValidation


K = 10
REPLICATES = 20
DATA_DIR = "datasets/Lab2.csv"
LOG_FILE = "results/lab_2_results.log"

logging.basicConfig(
    filename=LOG_FILE, filemode="w",
    format="%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",  # noqa
    datefmt="%H:%M:%S", level=logging.INFO
)


def perform_k_fold_cross_validation_with_regression(
    X, y, exp_name, model, performance_metric=sum_of_squared_error,
    k=10, replicates=20
):

    kfcv = KFoldCrossValidation(X=X, y=y, k=k)
    logging.info(f"Model: {exp_name}")
    scores = []
    for seed in range(replicates):
        performance = kfcv.get_cv_performance(
            model=model, performance_eval_func=performance_metric, seed=seed
        )
        scores.append(performance)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logging.info(f"Mean CV {performance_metric.__name__} across {replicates} replicates: {mean_score}")  # noqa
    logging.info(f"STD of CV {performance_metric.__name__} across {replicates} replicates: {std_score}")  # noqa
    logging.info("\n")


if __name__ == "__main__":

    # Read data
    loader = DataLoader(DATA_DIR)
    loader.load_data()
    norm_data, means, stds = loader.normalize_data(loader.data)  # normalize

    # Get correlation matrix
    correlation_matrix = loader.data.corr().round(2)
    logging.info(f"Correlation matrix: {correlation_matrix}")

    # Model
    linear_regression = LinearRegression(risk_function=sum_of_squared_error)

    # 8-predictor model
    X_eight = norm_data[["x1", "x2", "x3", "x4",
                         "x5", "x6", "x7", "x8"]].to_numpy()
    y_eight = norm_data["y"].to_numpy()
    #   CV
    perform_k_fold_cross_validation_with_regression(
        X_eight, y_eight, "8-Predictor Model", linear_regression,
        k=K, replicates=REPLICATES
    )

    # 2-predictor model
    X_two = norm_data[["x1", "x2"]].to_numpy()
    y_two = norm_data["y"].to_numpy()
    #   CV
    perform_k_fold_cross_validation_with_regression(
        X_two, y_two, "2-Predictor Model", linear_regression,
        k=K, replicates=REPLICATES
    )
