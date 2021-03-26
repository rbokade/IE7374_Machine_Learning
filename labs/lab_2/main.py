import logging

import numpy as np

from data_loader import DataLoader
from k_fold_cv import KFoldCrossValidation
from multi_linear_regression import MultiLinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Lab-2")


DATA_DIR = "datasets/Lab2.csv"
REPLICATES = 20

def compute_sse(y_pred, y):
    y = np.array(y)
    y_prep = np.array(y_pred)
    return sum((y - y_pred) ** 2)


if __name__ == "__main__":

    # Read data
    loader = DataLoader(DATA_DIR)
    loader.load_data()
    norm_data, means, stds = loader.normalize_data(loader.data)  # normalize

    # Model
    mlr_normal = MultiLinearRegression()

    # 8-predictor model
    #   CV
    cv_eight = KFoldCrossValidation(
        X=norm_data[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]].to_numpy(),
        y=norm_data["y"].to_numpy(), k=10
    )
    logger.info("Model: 8-predictor")
    #   Repeat experiment 20 times
    cv_eight_scores = []
    for seed in range(REPLICATES):
        performance = cv_eight.get_cv_performance(
            model=mlr_normal,
            performance_eval_func=compute_sse,
            seed=seed
        )
        cv_eight_scores.append(performance)
    mean_cv_eight_score = np.mean(cv_eight_scores)
    std_cv_eight_score = np.std(cv_eight_scores)
    logger.info(
        f"Mean CV score across {REPLICATES} replicates: {mean_cv_eight_score}"
    )
    logger.info(
        f"STD of CV score across {REPLICATES} replicates: {std_cv_eight_score}"
    )

    logger.info("\n\n")

    # 2-predictor model
    #   CV
    cv_two = KFoldCrossValidation(
        X=norm_data[["x1", "x2"]].to_numpy(),
        y=norm_data["y"].to_numpy(), k=10
    )
    logger.info("Model: 2-predictor")
    #   Repeat experiment 20 times
    cv_two_scores = []
    for seed in range(REPLICATES):
        performance = cv_two.get_cv_performance(
            model=mlr_normal,
            performance_eval_func=compute_sse,
            seed=seed
        )
        cv_two_scores.append(performance)
    mean_cv_two_score = np.mean(cv_two_scores)
    std_cv_two_score = np.std(cv_two_scores)
    logger.info(
        f"Mean CV score across {REPLICATES} replicates: {mean_cv_two_score}"
    )
    logger.info(
        f"STD of CV score across {REPLICATES} replicates: {std_cv_two_score}"
    )
