import logging

from math import ceil

import numpy as np


class KFoldCrossValidation:
    """
    - Divide the data into K parts of roughly equal size
    - Evaluation:
        - For "i"-th in "k" parts, set one part "j" aside
        - Train the model on remaining parts
        - Use this model to train on "j" part and evaluate the given
        performance metric (e.g., SSE_i)
        - The final cross-validation score is the average of
        the performance metrics (e.g. mean({SSE_{1}, ..., SSE_{k}))

    Parameters:
    -----------
        @parameter k: (int) Number of parts to divide the data into
        @parameter X: (np.ndarray) Values of independent variables
        @parameter y: (np.ndarray) Values of predictor variable
        @parameter seed: (int) Seed for reproducible results
    """

    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        assert self.k > 1, "k must be greater than 1."
        self.num_rows, self.num_cols = self.X.shape

    def _compute_num_samples_per_part(self):
        """
        Splits the data into k equal parts. If the data cannot be split into
        equal parts, then the last part would contain the remainder of the data
        """
        # Split equally
        q, r = divmod(self.num_rows, self.k)
        self.num_samples_per_part = [ceil(q + (r / self.k)) for _ in range(self.k - 1)]  # noqa
        remaining_samples = self.num_rows - sum(self.num_samples_per_part)
        self.num_samples_per_part.append(remaining_samples)  # add remaining samples  # noqa
        logging.debug(f"Number of samples per part: {self.num_samples_per_part}")  # noqa

        return self.num_samples_per_part

    def _get_split_indices(self):
        """
        Computes indices for each part. Storing indices in memory is less
        expensive than storing the data.
        """
        indices = np.arange(0, self.num_rows)

        self.split_indices = []
        for i in range(self.k - 1):
            # Get a random sample of indices for the part
            sampled_indices = np.random.choice(
                indices, size=self.num_samples_per_part[i], replace=False
            )
            self.split_indices.append(sampled_indices)
            # Remove the selected indices
            indices = np.setdiff1d(indices, sampled_indices)

        # Add the remaining indices in the last part
        self.split_indices.append(indices)
        self.split_indices = np.array(self.split_indices, dtype="object")

    def get_cv_performance(self, model, performance_eval_func, seed=42):
        """
        Performs k-fold cross validation of the dataset.

        Parameters:
        -----------
            @param model: (class) model with fit() and predict() methods
            @param performance_metric: (func) function to evaluate the
                performance of the model
            @param seed: (int) Seed
        """
        np.random.seed(seed)  # set random seed for reproducibility
        self._compute_num_samples_per_part()
        self._get_split_indices()

        # Holding out each part (j), training on remaining parts
        # and evaluating the performance on part (j)
        performances = []
        for holdout_idx in range(self.k):
            # Train
            indices_for_training = np.delete(
                self.split_indices, holdout_idx, axis=0
            )
            indices_for_training = np.concatenate(indices_for_training, axis=0)
            X = self.X[tuple(indices_for_training), :]
            if self.y.ndim == 1:
                y = self.y[indices_for_training]
            else:
                y = self.y[tuple(indices_for_training), :]
            _ = model.fit(X, y)
            # Evaluate
            indices_for_testing = self.split_indices[holdout_idx]
            X_test = self.X[indices_for_testing]
            y_test = self.y[indices_for_testing]
            y_predicted = model.predict(X_test)
            performance = performance_eval_func(
                y_actual=y_test, y_predicted=y_predicted
            )
            performances.append(performance)
            logging.debug(f"Holdout part: {holdout_idx} | {performance_eval_func.__name__}: {performance}")  # noqa

        mean_performance = np.mean(performances)
        logging.info(f"Seed: {seed} | Mean CV {performance_eval_func.__name__}: {mean_performance}")  # noqa

        return mean_performance
