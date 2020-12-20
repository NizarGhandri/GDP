import numpy as np
from scipy.stats import t
from scipy import stats
from itertools import chain, combinations


def correlation_test(X, threshold=0.95):
    """

    :param X:
    :param threshold:
    :return:
    """
    # doesn't work yet
    correlation_matrix = np.corrcoef(X)
    correlation_matrix < threshold
    return correlation_matrix


def confidence_interval(x_shape, variable, var, tolerance=0.95):
    """
    calculating confidence interval
    :param x_shape:
    :param variable:
    :param var:
    :param tolerance:
    :return:
    """
    # giving back a n x 2 matrix, on the first column lower bound, second column upper bound
    # x - explanatory variables
    # beta - parameteres
    # var - calculated variance (can differ dependent on model)
    # tolerance - which tolerance level do you want

    n, m = x_shape
    degoffree = n - m
    alpha = stats.t.ppf(1 - tolerance / 2, degoffree) * np.sqrt(var)
    return variable - alpha, variable + alpha


def standard_error_regression(y, y_hat, X_shape):
    """
    calculating Standard Error
    :param y:
    :param y_hat:
    :param X_shape:
    :return:
    """
    e = y - y_hat
    return (e.T @ e) / (X_shape[0] - X_shape[1])


def variance_least_squares_weights(y, y_hat, X):
    """
    calculating variance of least squares
    :param y:
    :param y_hat:
    :param X:
    :return:
    """
    return standard_error_regression(y, y_hat, X.shape) * np.reshape(np.diag(np.linalg.inv(X.T @ X)), (-1, 1))


# def variance_least_squares_line (y, y_hat, X):
#     var = (X - np.mean(X, axis=0))**2
#     return (var/sum(var) + 1/X.shape[0]) * standard_error_regression (y, y_hat, X.shape)


def variance_least_squares_line(y, y_hat, X):
    """
    calculating variance of least squares
    :param y:
    :param y_hat:
    :param X:
    :return:
    """
    return standard_error_regression(y, y_hat, X.shape) * (
        np.reshape(np.diag(X @ np.linalg.inv(X.T @ X) @ X.T), (-1, 1)))


def subset_iterator(X_columns):
    """
    To use this to find a all the subsets of X you do the following: 
    for columns in subset_iterator(range(X.shape[1])):
        X[:, columns] # this will be your new dataset out of the subsets 

    :param X_columns:
    :return:
    """
    rnge = range(X_columns)
    return chain(*map(lambda x: combinations(rnge, x), range(2, X_columns + 1)))
