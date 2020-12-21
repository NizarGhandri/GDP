import numpy as np


def least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Least Squares Estimator. We use np.solve for more numerical stability.

    :param X: training features
    :param y: training labels
    :return: least squares weights
    """
    X_t = X.T
    return np.linalg.solve(X_t.dot(X), X_t.dot(y))


def ridge_regression(X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Ridge regression using normal equations.

    :param X: training features
    :param y: training lables
    :param lambda_: penalty term
    :return: ridge regression weights
    """
    x_transpose = X.T
    left = x_transpose.dot(X) + lambda_ * np.identity(X.shape[1])
    right = x_transpose.dot(y)
    w = np.linalg.solve(left, right)
    return w
