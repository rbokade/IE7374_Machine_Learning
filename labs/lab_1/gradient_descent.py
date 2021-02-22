import numpy as np
np.random.seed(42)

from multi_linear_regression import MultiLinearRegression, cost_fn


class MLR_GradientDescent(MultiLinearRegression):
    def __init__(self, risk_fn=None, thetas=None, learning_rate=4e-4, tolerance=5e-3):
        MultiLinearRegression.__init__(self, risk_fn, thetas, learning_rate, tolerance)
        self._cost_history = []

    def fit(self, X, y, max_iter=50000):
        assert (isinstance(max_iter, int) and max_iter >= 1), \
                "max_iter {} should be integer and greater than 1.".format(max_iter)
        if not isinstance(X, np.ndarray): X = np.asarray(X)
        if not isinstance(y, np.ndarray): y = np.asarray(y)
        assert X.shape[0] == y.shape[0], "Input samples should be consistent."

        X = self._add_constants_col(X)
        if self._thetas is None:
            self._thetas = np.zeros(X.shape[1])

        m = X.shape[0]
        self._cost_history = []
        prev_rmse = 0
        for i in range(max_iter):
            error = self._risk_fn(X, y, self._thetas)
            gradient = 1 / m * np.dot(X.T, error)
            self._thetas -= self._lr * gradient
            sq_error = np.sum(np.square(error))
            rmse = np.sqrt(sq_error / m)
            # cost_history.append(sq_error / (2 * m))
            self._cost_history.append(rmse)
            grad = np.sum(gradient)
            # print("gradient: {} | error: {} | rmse: {}".format(grad, sq_error, rmse))
            if np.abs(prev_rmse - rmse) <= self._tol:
                break
            prev_rmse = rmse
        print("\n\n")
        self.print_performance(X, y)
        return self._thetas

    def get_cost_history(self):
        return np.asarray(cost_history)


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
    vgd = MLR_GradientDescent(tolerance=5e-10)
    print("coeffs", vgd.fit(X, y))
    norm_test, _, _ = loader.normalize_data(test, means, stds)
    print("test performance", vgd.evaluate(norm_test.iloc[:, :-1], norm_test.iloc[:, -1]))
