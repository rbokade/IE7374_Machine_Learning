import numpy as np


def compute_correlation_matrix(X, y=None):
    return np.corrcoef(X, y)
