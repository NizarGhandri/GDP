import numpy as np


def SST(y: np.ndarray) -> float:
    """
    Computes the deviations of from its mean Total Sum of Squares.
    Used to judge the goodness to fit of the feature to predict.

    :param y: real data
    :return: Total Sum of Squares
    """
    y_avg = np.mean(y)
    return float(np.sum(np.square(y - y_avg)))


def M_zero(n: int) -> np.ndarray:
    """
    Computes the M_0 matrix.
    M_0 * x = x - x_bar

    :param n: dimension of the matrix
    :return: squared matrix M_0
    """
    return np.eye(n) - np.ones((n, n)) / n


def SSR(y_hat: np.ndarray) -> float:
    """
    Computes the regression sum of squares.

    :param y_hat: data matrix
    :return: regression sum of squares
    """
    n = y_hat.shape[0]
    return float(y_hat.T @ M_zero(n) @ y_hat)


def SSE(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Computes the sum of squared errors.

    :param y: real data
    :param y_hat: prediction
    :return: sum of squares errors
    """
    e = y - y_hat
    return float(np.sum(e * e))


def R_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Computes the coefficient of determination (R²).

    :param y: real data
    :param y_hat: prediction
    :return: coefficient of determination (R²)
    """
    return 1 - SSE(y, y_hat) / SST(y)


def adjusted_R_squared(y: np.ndarray, y_hat: np.ndarray, n_features: int) -> float:
    """
    Computes the adjusted coefficient of determination (R²).

    :param y: real data
    :param y_hat: prediction
    :param n_features: number of features (K)
    :return: adjusted coefficient of determination (R²)
    """
    n = len(y)
    assert(n > n_features + 1)
    r_squared = R_squared(y, y_hat)
    return 1 - (1 - r_squared) * (n - 1) / (n - n_features - 1)
