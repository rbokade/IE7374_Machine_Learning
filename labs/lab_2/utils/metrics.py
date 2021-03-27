import numpy as np


def squared_error(y_actual, y_predicted):
    return np.power(y_actual - y_predicted, 2)


def sum_of_squared_error(y_actual, y_predicted):
    return np.sum(squared_error(y_actual, y_predicted))


def mean_squared_error(y_actual, y_predicted):
    return np.mean(squared_error(y_actual, y_predicted))


def root_mean_squared_error(y_actual, y_predicted):
    return np.sqrt(mean_squared_error(y_actual, y_predicted))
