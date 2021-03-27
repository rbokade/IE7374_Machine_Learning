

class BaseLearner:

    def _init_risk_fn(self):
        raise NotImplementedError

    def _add_constants_col(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def print_performance(self):
        raise NotImplementedError
