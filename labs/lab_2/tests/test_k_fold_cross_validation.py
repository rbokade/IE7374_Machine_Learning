import sys
import unittest
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np

from utils.k_fold_cross_validation import KFoldCrossValidation


class SampleModel:

    def fit(self, X, y):
        return y

    def predict(self, X):
        return np.ones(X.shape[0])


def compute_total_abs_error(y_actual, y_predicted):
    return np.sum(np.abs(y_actual - y_predicted))


class TestKFoldCrossValidation(unittest.TestCase):

    def test_splits(self):
        X = np.arange(0, 790).reshape((79, 10))
        y = np.ones(X.shape[0])
        # 8 obs in first 9 parts and 7 in the last
        kfcv = KFoldCrossValidation(X, y, k=10)
        kfcv._compute_num_samples_per_part()
        assert kfcv.num_samples_per_part == [8] * 9 + [7]

    def test_cross_validation(self):
        model = SampleModel()
        X = np.arange(0, 790).reshape((79, 10))
        y = np.ones(X.shape[0])
        # 8 obs in first 9 parts and 7 in the last
        kfcv = KFoldCrossValidation(X, y, k=10)
        for s in range(20):
            kfcv_performance = kfcv.get_cv_performance(
                model, compute_total_abs_error, seed=s
            )
            assert kfcv_performance == 0


if __name__ == "__main__":

    unittest.main()
