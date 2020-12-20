import numpy as np


def least_squares(X, y):
    """
    Compute least squares with np.solve for more numerical stability
    :param X:
    :param y:
    :return:
    """
    X_t = X.T
    return np.linalg.solve(X_t.dot(X), X_t.dot(y))


def ridge_regression(tx, y, lambda_):
    """
    Ridge regression using normal equations
    :param tx:
    :param y:
    :param lambda_:
    :return:
    """
    x_transpose = tx.T
    left = x_transpose.dot(tx) + lambda_ * np.identity(tx.shape[1])
    right = x_transpose.dot(y)
    w = np.linalg.solve(left, right)
    return w
