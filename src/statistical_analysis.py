import copy
from itertools import chain, combinations
from typing import Tuple, List

import numpy as np
from numpy import linalg
from scipy import stats
from scipy.stats import chi2
from regressions import least_squares
from helpers import *
from evaluation_metrics import *


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


def confidence_interval(n: int, k: int, variable: np.ndarray, variance: np.ndarray,
                        percentage: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the confidence interval.
    
    :param n: number of datapoints
    :param k: number of features
    :param variable: variable for which the CI is computed
    :param variance: variance of the variable
    :param percentage: percentage of the CI
    :return: CI's lower and upper bounds
    """
    deg_of_freedom = n - k
    alpha = stats.t.ppf(1 - (1 - percentage) / 2, deg_of_freedom) * np.sqrt(variance)
    return variable - alpha, variable + alpha


def standard_error_regression(y: np.ndarray, y_hat: np.ndarray, k: int) -> float:
    """
    Computes the Standard Error.

    :param y: real data
    :param y_hat: prediction
    :param k: number of features
    :return: standard error
    """
    n = len(y)
    return SSE(y, y_hat) / (n - k)


def variance_least_squares_weights(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Computes the variance of least squares.

    :param X: features
    :param y: labels
    :param y_hat: predictions
    :return: array of variances for the weights of the regression
    """
    return standard_error_regression(y, y_hat, X.shape[1]) * np.reshape(np.diag(np.linalg.inv(X.T @ X)), (-1, 1))


# def variance_least_squares_line (y, y_hat, X):
#     var = (X - np.mean(X, axis=0))**2
#     return (var/sum(var) + 1/X.shape[0]) * standard_error_regression (y, y_hat, X.shape)


def variance_least_squares_line(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Computes the variance of least squares.

    :param X: features
    :param y: labels
    :param y_hat: predictions
    :return: array of variances for the predictd labels
    """
    return standard_error_regression(y, y_hat, X.shape[1]) * (
        np.reshape(np.diag(X @ np.linalg.inv(X.T @ X) @ X.T), (-1, 1)))


def subset_iterator(n_features):
    """
    To use this to find a all the subsets of X you do the following: 
    for columns in subset_iterator(range(X.shape[1])):
        X[:, columns] # this will be your new dataset out of the subsets 

    :param n_features: number of features
    :return: all possible combinations of numbers from 0 to n_features
    """
    rnge = range(n_features)
    return chain(*map(lambda x: combinations(rnge, x), range(2, n_features + 1)))


def ttest(X_shape, betak, vark, tolerance=0.95) -> bool:
    """
    Computes the statistical significance of a specific variable.

    :param X_shape: shape of the observed matrix
    :param betak: estimator of the specific parameter
    :param vark: variance of specific parameter
    :param tolerance: the tolerance for the tolerance interval
    :return: true if it is statistically significant, false if it is not
    """

    n, m = X_shape
    degoffree = n - m

    # computes t-tst
    tt = stats.t.ppf(1 - (1 - tolerance) / 2, degoffree)
    tk = betak / np.sqrt(vark)

    if tk > tt:
        test = True
    else:
        test = False

    return test


def breusch_pagan_test(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, str]:
    """
    taken from:
    https://stackoverflow.com/questions/30061054/ols-breusch-pagan-test-in-python
    
    Breusch-Pagan test for heteroskedasticity in a linear regression model:
    H_0 = No heteroskedasticity.
    H_1 = Heteroskedasticity is present.
    
    :param X: features
    :param y: labels
    :return: Breusch-Pagan test statistic, the p-value for the test and the test result.
    """
    y = y.ravel()

    if y.ndim != 1:
        raise SystemExit('Error: y has more than 1 dimension.')
    if X.shape[0] != y.shape[0]:
        raise SystemExit('Error: the number of samples differs between x and y.')
    else:
        n_samples = y.shape[0]

    # fit an OLS linear model to y using x:
    w = least_squares(X, y)

    # calculate the squared errors:
    err = (y - predict(X, w)) ** 2

    # fit an auxiliary regression to the squared errors:
    # why?: to estimate the variance in err explained by x
    w_aux = least_squares(X, err)
    pred_err = predict(X, w_aux)

    # calculate the coefficient of determination:
    r2 = R_squared(err, pred_err)

    # calculate the Lagrange multiplier:
    LM = n_samples * r2

    # calculate p-value
    pval = chi2.sf(LM, X.shape[1])

    if pval < 0.05:
        test_result = 'Heteroskedasticity present at 95% CI.'
    else:
        test_result = 'No significant heteroskedasticity.'

    return LM, pval, test_result


def condition_number(X: np.ndarray) -> float:
    """
    Computes the Condition Number. The bigger it is, the worse the multicolinearity, starts to become a problem from
    20 on.
    
    :param X: Observed matrix
    :return: condition number
    """
    w, v = linalg.eig(np.dot(np.transpose(X), X))
    return np.sqrt(np.max(w) / np.min(w))


def VIF(X: np.ndarray) -> np.ndarray:
    """
    Computes the Variance Inflation Factor, the bigger the worse the multicolinearity.
    
    :param X: Observed matrix
    :return: VIF
    """

    xtemp2 = copy.copy(X)

    n_features = X.shape[1]

    VIFF = np.zeros(n_features)

    for i in range(n_features):
        indices = list(range(n_features))
        indices.remove(i)

        x0 = xtemp2[:, indices]
        y0 = xtemp2[:, i]

        beta = least_squares(x0, y0)
        y_hat = predict(x0, beta)

        VIFF[i] = 1 / (1 - R_squared(y0, y_hat))

    return VIFF


def general_to_simple(X: np.ndarray, y: np.ndarray) -> List[int]:
    """
    Finds the relevant features using the general to simple approach.

    :param X: The matrix of observables
    :param y: The outcome matrix
    :return: list of indices
    """
    n, k = np.shape(X)

    indices = list(range(k))

    ttest_result = False

    while (not ttest_result) and len(indices) > 1:

        X_temp = np.copy(X[:, indices])

        index_to_delete = indices[0]

        r_2 = -math.inf

        for i in indices:

            new_indices = list(np.copy(indices))
            new_indices.remove(i)

            x0 = X_temp[:, new_indices]

            beta_reduced = least_squares(x0, y)
            y_hat_reduced = predict(x0, beta_reduced)
            r = R_squared(y, y_hat_reduced)
            if r > r_2:
                index_to_delete = i
                r_2 = r

        beta = least_squares(X_temp, y)
        y_hat = predict(X_temp, beta)

        var = variance_least_squares_weights(X_temp, y, y_hat)

        ttest_result = ttest(np.shape(X_temp), beta[index_to_delete], var[index_to_delete],
                             tolerance=0.95)

        if not ttest_result:
            indices.remove(index_to_delete)

    return indices


# def simple_to_general(X, y):
#     """
#     Computes the model from the simple to general approach.
#
#     :param X: The matrix of observables
#     :param y: The outcome matrix
#     :return: model from the simple to general approach
#     """
#
#     shape = np.shape(X)
#     xtemp2 = copy.copy(X)
#     a = np.zeros(shape[1])
#
#     for f in range(shape[1] - 1):
#         for i in range(shape[1] - f):
#             if f == 0:
#                 x0 = xtemp2[:, i]
#                 x0 = x0.transpose()
#                 x0 = np.expand_dims(x0, axis=1)
#                 beta_reduced = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x0), x0)), np.transpose(x0)), y)
#                 y_hat_reduced = np.dot(x0, beta_reduced)
#                 a[i] = R_squared(y, y_hat_reduced)
#
#             else:
#                 x1 = xtemp2[:, i]
#                 x1 = np.expand_dims(x1, axis=1)
#                 x2 = np.hstack((x0, x1))
#                 beta_reduced = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x2), x2)), np.transpose(x2)), y)
#                 y_hat_reduced = np.dot(x2, beta_reduced)
#                 a[i] = R_squared(y, y_hat_reduced)
#         if f == 0:
#             stat_sign = False
#             ind = np.argmax(a[:])
#             x0 = xtemp2[:, ind]
#             x0 = np.expand_dims(x0, axis=1)
#
#         else:
#             ind = np.argmax(a[:])
#             x1 = xtemp2[:, ind]
#             x1 = np.expand_dims(x1, axis=1)
#             x2 = np.hstack((x0, x1))
#             beta_full = least_squares(x2, y)
#             y_hat_full = np.dot(x2, beta_full)
#             shapex2 = np.shape(x2)
#             error = np.dot(np.transpose(y - np.dot(x2, beta_full)), y - np.dot(x2, beta_full)) / (
#                     shapex2[0] - shapex2[1])
#             var = np.linalg.inv(np.dot(np.transpose(x2), x2)) * error
#             stat_sign = ttest(np.shape(x2), beta_full[f, 0], var[f, f], tolerance=0.95)
#             del x1, beta_full, y_hat_full, var, error, shapex2
#
#         if stat_sign:
#             return x2
#         else:
#             xtemp2 = np.delete(xtemp2, ind, axis=1)
#             if f == 0:
#                 r = 1
#                 a = np.zeros(shape[1])
#             else:
#                 x0 = x2
#                 a = np.zeros(shape[1])
#
#     return x0
#

def simple_to_general(X: np.ndarray, y: np.ndarray) -> List[int]:
    """
    Finds the relevant features using the simple to general approach.

    :param X: The matrix of observables
    :param y: The outcome matrix
    :return: list of indices
    """

    n, k = np.shape(X)

    indices = []
    remaining_indices = list(range(k))

    ttest_result = False

    while ttest_result and len(indices) < k:

        index_to_add = indices[0]

        r_2 = -math.inf

        for i in remaining_indices:

            new_indices = list(np.copy(indices))
            new_indices.append(i)

            x0 = X[:, new_indices]

            beta_augmented = least_squares(x0, y)
            y_hat_augmented = predict(x0, beta_augmented)
            r = R_squared(y, y_hat_augmented)
            if r > r_2:
                index_to_add = i
                r_2 = r

        indices.append(index_to_add)

        if len(indices) > 1:
            X_temp = np.copy(X[:, indices])

            beta = least_squares(X_temp, y)
            y_hat = predict(X_temp, beta)

            var = variance_least_squares_weights(X_temp, y, y_hat)

            ttest_result = ttest(np.shape(X_temp), beta[index_to_delete], var[index_to_delete],
                                 tolerance=0.95)

            if not ttest_result:
                # index to add is actually not relevant, delete it
                indices.remove(index_to_add)

    return indices
