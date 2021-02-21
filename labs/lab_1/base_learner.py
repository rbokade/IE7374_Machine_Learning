

class BaseLearner:
    def __init__(self, risk_fn=None, thetas=None, learning_rate=None, tolerance=None):
        self._thetas = thetas
        self._risk_fn = risk_fn
        self._lr = learning_rate
        self._tol = tolerance
        self._init_risk_fn()

    def _init_risk_fn(self):
        raise NotImplementedError

    def _add_constants_col(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(X):
        raise NotImplementedError

    def print_performance(self):
        raise NotImplementedError

