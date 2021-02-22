import os

import yaml
import numpy as np

from data_loader import DataLoader
from multi_linear_regression import MultiLinearRegression
from gradient_descent import MLR_GradientDescent


def train(configs, save_results=True):
    results = {dataset: {} for dataset in datasets}
    for dataset in datasets:
        results[dataset]["mlr_normal"] = {}
        results[dataset]["mlr_gd"] = {}
        # Load data
        loader = DataLoader(configs[dataset]["data_dir"])
        loader.load_data()
        train, test = loader.split_dataset_into_train_test()
        norm_train, means, stds = loader.normalize_data(train)  # normalize
        norm_test, _, _ = loader.normalize_data(test, means, stds) # normalize test set based on train
        # Train
        #   MLR Normal
        mlr_normal = MultiLinearRegression()
        results[dataset]["mlr_normal"]["coefficients"] = mlr_normal.fit(
                norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
        ).tolist()
        results[dataset]["mlr_normal"]["train_rmse"] = mlr_normal.evaluate(
                norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
        )
        #       Performance on testing set
        results[dataset]["mlr_normal"]["test_rmse"] = mlr_normal.evaluate(
                norm_test.iloc[:, :-1], norm_test.iloc[:, -1]
        )
        #   MLR GD
        mlr_gd = MLR_GradientDescent(
            learning_rate=configs[dataset]["learning_rates"][0],
            tolerance=configs[dataset]["tolerances"][0],
        )
        results[dataset]["mlr_gd"]["coefficients"] = mlr_gd.fit(
                norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
        ).tolist()
        results[dataset]["mlr_gd"]["train_rmse"] = mlr_gd.evaluate(
                norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
        )
        #       Performance on testing set
        norm_test, _, _ = loader.normalize_data(test, means, stds)
        results[dataset]["mlr_gd"]["test_rmse"] = mlr_gd.evaluate(
                norm_test.iloc[:, :-1], norm_test.iloc[:, -1]
        )
    if save_results:
        with open("results_with_given_tolerance.yaml", "w") as f:
            yaml.dump(results, f)


def try_out_all_tolerances(configs, save_results=True):
    results = {dataset: {} for dataset in datasets}
    for dataset in datasets:
        results[dataset]["mlr_normal"] = {}
        results[dataset]["mlr_gd"] = {
                "tolerance_{}".format(tolerance): {}
                for tolerance in configs[dataset]["tolerances"]
        }
        # Load data
        loader = DataLoader(configs[dataset]["data_dir"])
        loader.load_data()
        train, test = loader.split_dataset_into_train_test()
        norm_train, means, stds = loader.normalize_data(train)  # normalize
        norm_test, _, _ = loader.normalize_data(test, means, stds) # normalize test set based on train
        # Train
        #   MLR Normal
        mlr_normal = MultiLinearRegression()
        results[dataset]["mlr_normal"]["coefficients"] = mlr_normal.fit(
                norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
        ).tolist()
        results[dataset]["mlr_normal"]["train_rmse"] = mlr_normal.evaluate(
                norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
        )
        #       Performance on testing set
        results[dataset]["mlr_normal"]["test_rmse"] = mlr_normal.evaluate(
                norm_test.iloc[:, :-1], norm_test.iloc[:, -1]
        )
        #   MLR GD
        for tolerance in configs[dataset]["tolerances"]:
            tolerance_key = "tolerance_{}".format(tolerance)

            mlr_gd = MLR_GradientDescent(
                learning_rate=configs[dataset]["learning_rates"][0],
                tolerance=tolerance,
            )
            results[dataset]["mlr_gd"][tolerance_key]["coefficients"] = mlr_gd.fit(
                    norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
            ).tolist()
            results[dataset]["mlr_gd"][tolerance_key]["train_rmse"] = mlr_gd.evaluate(
                    norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
            )
            #       Performance on testing set
            norm_test, _, _ = loader.normalize_data(test, means, stds)
            results[dataset]["mlr_gd"][tolerance_key]["test_rmse"] = mlr_gd.evaluate(
                    norm_test.iloc[:, :-1], norm_test.iloc[:, -1]
            )
    if save_results:
        with open("results_with_all_tolerances.yaml", "w") as f:
            yaml.dump(results, f)


if __name__ == "__main__":

    # Configs
    datasets = ["yachtData", "concreteData", "housing"]
    configs = {dataset: {} for dataset in datasets}

    configs["housing"]["data_dir"] = os.path.abspath("datasets/housing.csv")
    configs["housing"]["learning_rates"] = [4e-4]
    configs["housing"]["tolerances"] = [5e-3, 0.1, 0.01, 0.05, 5e-10]

    configs["yachtData"]["data_dir"] = os.path.abspath("datasets/yachtData.csv")
    configs["yachtData"]["learning_rates"] = [1e-3]
    configs["yachtData"]["tolerances"] = [1e-3, 0.1, 0.01, 0.05, 5e-10]

    configs["concreteData"]["data_dir"] = os.path.abspath("datasets/concreteData.csv")
    configs["concreteData"]["learning_rates"] = [7e-4]
    configs["concreteData"]["tolerances"] = [1e-4, 0.1, 0.01, 0.05, 5e-10]

    train(configs)
    try_out_all_tolerances(configs)
