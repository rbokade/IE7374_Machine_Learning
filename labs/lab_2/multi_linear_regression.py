import logging
import numpy as np

from base_learner import BaseLearner


def cost_fn(X, y, theta):
    y_est = np.dot(X, theta)
    return y_est.squeeze() - y.squeeze()


class MultiLinearRegression(BaseLearner):
    _logger = logging.getLogger(__name__)

    def __init__(self, risk_fn=None, thetas=None, learning_rate=None, tolerance=None):
        BaseLearner.__init__(self, risk_fn, thetas, learning_rate, tolerance)

    def _init_risk_fn(self):
        if self._risk_fn is None:
            self._risk_fn = cost_fn

    def _add_constants_col(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X)) # add constants column

    def print_performance(self, X, y):
        errors = self._risk_fn(X, y, self._thetas)
        rmse = np.sqrt(np.mean(errors ** 2))
        self._logger.debug("Intercept: {}".format(self._thetas[0]))
        self._logger.debug("Coefficients: {}".format(" ".join(str(theta) for theta in self._thetas[1:]), sep=" ", end=" "))
        self._logger.debug("RMSE: {}".format(rmse))
        self._logger.debug("\n")

    def fit(self, X, y):
        if not isinstance(X, np.ndarray): X = np.asarray(X)
        if not isinstance(y, np.ndarray): y = np.asarray(y)
        assert X.shape[0] == y.shape[0], "Input samples should be consistent in size."
        X = self._add_constants_col(X)
        self._thetas = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.print_performance(X, y)
        return self._thetas

    def predict(self, X):
        X = self._add_constants_col(X)
        return np.dot(X, self._thetas)

    def evaluate(self, X, y):
        if not isinstance(X, np.ndarray): X = np.asarray(X)
        if not isinstance(y, np.ndarray): y = np.asarray(y)
        assert X.shape[0] == y.shape[0], "Input samples should be consistent in size."
        X = self._add_constants_col(X)
        errors = self._risk_fn(X, y, self._thetas)
        rmse = np.sqrt(np.mean(errors ** 2))
        return rmse.item()


if __name__ == "__main__":

    # Testing
    from data_loader import DataLoader
    # Load data
    data_dir = "datasets/yachtData.csv"
    loader = DataLoader(data_dir)
    loader.load_data()
    train, test = loader.split_dataset_into_train_test()
    norm_train, means, stds = loader.normalize_data(train)
    # Gradient Descent
    X, y = norm_train.iloc[:, :-1], norm_train.iloc[:, -1]
    mlr = MultiLinearRegression()
    self._logger.info("coeffs", mlr.fit(X, y))
    norm_test, _, _ = loader.normalize_data(test, means, stds)
    self._logger.info("test performance", mlr.evaluate(norm_test.iloc[:, :-1], norm_test.iloc[:, -1]))

