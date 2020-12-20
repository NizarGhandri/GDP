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
    assert (n > n_features + 1)
    r_squared = R_squared(y, y_hat)
    return 1 - (1 - r_squared) * (n - 1) / (n - n_features - 1)


def MSE(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the Mean Square Error.

    :param y: real data
    :param y_hat: prediction
    :return: MSE
    """
    return SSE(y, y_hat) / len(y)


def RMSE(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the Root Mean Square Error.

    :param y: real data
    :param y_hat: prediction
    :return: RMSE
    """
    return np.sqrt(MSE(y, y_hat))


def MAE(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the Mean Absolute Error.

    :param y: real data
    :param y_hat: prediction
    :return: MAE
    """
    e = y - y_hat
    return np.sum(np.abs(e)) / len(e)


def theil_U(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Compute the Theil U statistic (scale invariant measure).

    :param y: real data
    :param y_hat: prediction
    :return: Theil U statistic
    """
    return np.sqrt(MSE(y, y_hat) / MSE(y, np.zeros_like(y)))


def information_criteria(y: np.ndarray, y_hat: np.ndarray, n_features: int, ic_type=None) -> float:
    if ic_type is None:
        raise ValueError("Information Criteria type not defined. `ic_type` is None, must be 'akaike' or 'bayesian'.")
    e = y - y_hat
    n = len(e)
    if ic_type == 'akaike':
        return np.log(e @ e.T / n) + 2 * n_features / n
    elif ic_type == 'bayesian':
        return np.log(e @ e.T / n) + n_features * np.log(n) / n
    else:
        raiseValueError("Invalid Information Criteria type. `ic_type` must be 'akaike' or 'bayesian'.")
