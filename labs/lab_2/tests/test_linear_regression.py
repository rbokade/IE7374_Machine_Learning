import sys
import unittest
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import numpy as np

from learners.linear_regression import LinearRegression


class TestLinearRegression(unittest.TestCase):

    def test_regression(self):
        X, y = np.random.rand(10, 10), np.zeros((10, 1))
        linear_regression = LinearRegression()
        thetas = linear_regression.fit(X, y)
        for theta in thetas:
            assert theta == 0
        assert linear_regression.evaluate(X, y) == 0


if __name__ == "__main__":

    unittest.main()
