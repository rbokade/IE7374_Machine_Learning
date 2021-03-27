import logging

import numpy as np

from learners.base_learner import BaseLearner
from utils.metrics import mean_squared_error


class LinearRegression(BaseLearner):
    def __init__(self, risk_function=None, initial_weights=None):
        BaseLearner.__init__(self)
        self._weights = initial_weights
        self._risk_function = risk_function
        self._init_risk_function()

    def _init_risk_function(self):
        if self._risk_function is None:
            self._risk_function = mean_squared_error

    def _add_constants_column(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))  # add intercept

    def _parse_inputs(self, X, y=None):
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
            assert X.shape[0] == y.shape[0], "Input samples should be consistent in size."  # noqa
            logging.debug(f"Input shapes:: X: {X.shape} | y: {y.shape[0]}")
        else:
            logging.debug(f"Input shapes:: X: {X.shape}")
        X = self._add_constants_column(X)
        return X, y

    def print_performance(self, X, y):
        y_predicted = self.predict(X, parse_inputs=False)
        errors = self._risk_function(y, y_predicted)
        thetas_str = " ".join(str(theta) for theta in self._weights[1:])
        logging.debug(f"Intercept: {self._weights[0]}")
        logging.debug(f"Coefficients: {thetas_str}")
        logging.debug(f"{self._risk_function.__name__}: {errors}")

    def fit(self, X, y):
        X, y = self._parse_inputs(X, y)
        self._weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.print_performance(X, y)
        return self._weights

    def predict(self, X, parse_inputs=True):
        if parse_inputs:
            X, _ = self._parse_inputs(X)
        return np.dot(X, self._weights)

    def evaluate(self, X, y):
        y_predicted = self.predict(X)
        return self._risk_function(y, y_predicted)


class GradientDescentLinearRegression(LinearRegression):
    def __init__(
        self, risk_function=None, initial_weights=None,
        learning_rate=None, tolerance=None
    ):
        super(GradientDescentLinearRegression, self).__init__(
            risk_function, initial_weights
        )
        self._learning_rate = learning_rate
        self._tolerance = tolerance
        self._cost_history = []

    def fit(self, X, y, max_iter=50000):

        assert (isinstance(max_iter, int) and max_iter >= 1), \
            f"max_iter {max_iter} should be integer and greater than 1."

        X, y = self._parse_inputs(X, y)

        if self._weights is None:
            self._weights = np.zeros(X.shape[1])

        previous_error = 0
        for iter in range(max_iter):
            gradient = self._compute_gradient(X, y)
            self._weights -= self._learning_rate * gradient
            error = self._risk_function(X, y, self._weights)
            self._cost_history.append(error)
            logging.debug(f"Iter: {iter} | Gradient: {sum(gradient)}")
            logging.info(f"Iter: {iter} | Error: {error}")
            if np.abs(previous_error - error) <= self._tolerance:
                logging.info(f"Converged at {iter}-th iteration")
                break
            previous_error = error
        self.print_performance(X, y)
        return self._weights

    def _compute_gradient(self, X, y):
        y_predicted = np.dot(X.T, self._weights)
        loss = y_predicted - y
        return np.dot(X.T, loss) / X.shape[0]

    @property
    def cost_history(self):
        return np.asarray(self._cost_history)
