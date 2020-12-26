from itertools import chain, combinations
from typing import List, Tuple

from numpy import linalg
from scipy import stats
from scipy.stats import chi2

from src.regressions import least_squares, ridge_regression
from src.evaluation_metrics import *
from src.helpers import *
import math


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


def variance_least_squares_line(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Computes the variance of least squares.

    :param X: features
    :param y: labels
    :param y_hat: predictions
    :return: array of variances for the predictd labels
    """
    return standard_error_regression(y, y_hat, X.shape[1]) * (1 + 
        np.reshape(np.diag(X @ np.linalg.inv(X.T @ X) @ X.T), (-1, 1)))


def subset_iterator(n_features: int):
    """
    To use this to find a all the subsets of X you do the following:
    for columns in subset_iterator(range(X.shape[1])):
        X[:, columns] # this will be your new dataset out of the subsets

    :param n_features: number of features
    :return: all possible combinations of numbers from 0 to n_features
    """
    rnge = range(n_features)
    return chain(*map(lambda x: combinations(rnge, x), range(3, n_features + 1)))


def best_subset_ls(X: np.ndarray, y: np.ndarray):
    """
    Computes the best subset of features.

    :param X: features
    :param y: labels
    :return: the scores of all subsets and best subset of features
    """
    scores = []
    subsets = []
    for i in subset_iterator(X.shape[1]):
        X_train, X_test, y_train, y_test = train_test_split(X[:, i], y, proportion=0.9, shuffle=False)
        ws = least_squares(X_train, y_train)
        scores.append(R_squared(y_test, predict(X_test, ws)))
        subsets.append(i)

    return scores, subsets


def ttest(X_shape: Tuple[int, int], betak: float, vark: float, tolerance: float = 0.95) -> bool:
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

    # computes t-test
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

    xtemp2 = np.copy(X)

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

    # list of features
    indices = list(range(k))

    # ttest_result encloses the relevance of the feature in question
    ttest_result = False

    # keep on deleting features while they are not relevant and there are still more than 1 feature remaining
    while (not ttest_result) and len(indices) > 1:

        # initialize the candidate feature to be removed and its r squared
        index_to_delete = indices[0]
        r_2 = -math.inf

        # find feature whose removal yields the largest r_square
        for i in indices:
            new_indices = list(np.copy(indices))
            new_indices.remove(i)

            x0 = X[:, new_indices]

            beta_reduced = least_squares(x0, y)

            y_hat_reduced = predict(x0, beta_reduced)
            r = R_squared(y, y_hat_reduced)
            if r > r_2:
                index_to_delete = i
                r_2 = r

        # keep only the features in indices
        X_temp = np.copy(X[:, indices])

        # test the relevance of the feature to be removed
        beta = least_squares(X_temp, y)
        y_hat = predict(X_temp, beta)

        var = variance_least_squares_weights(X_temp, y, y_hat)

        ttest_result = ttest(np.shape(X_temp), beta[indices.index(index_to_delete)],
                             var[indices.index(index_to_delete)], tolerance=0.95)

        # if the feature is irrelevant, remove it from indices
        if not ttest_result:
            indices.remove(index_to_delete)

    return indices


def general_to_simple_ridge(X: np.ndarray, y: np.ndarray) -> List[int]:
    """
    Finds the relevant features using the general to simple approach.

    :param X: The matrix of observables
    :param y: The outcome matrix
    :return: list of indices
    """
    n, k = np.shape(X)

    # list of features
    indices = list(range(k))

    # ttest_result encloses the relevance of the feature in question
    ttest_result = False

    # keep on deleting features while they are not relevant and there are still more than 1 feature remaining
    while (not ttest_result) and len(indices) > 1:

        # initialize the candidate feature to be removed and its r squared
        index_to_delete = indices[0]
        r_2 = -math.inf

        # find feature whose removal yields the largest r_square
        for i in indices:
            new_indices = list(np.copy(indices))
            new_indices.remove(i)

            x0 = X[:, new_indices]

            beta_reduced = least_squares(x0, y)

            y_hat_reduced = predict(x0, beta_reduced)
            r = R_squared(y, y_hat_reduced)
            if r > r_2:
                index_to_delete = i
                r_2 = r

        # keep only the features in indices
        X_temp = np.copy(X[:, indices])
        
        #cv
        lambda_, _ = cross_val_ridge(X_temp, y, max_degree=0)

        # test the relevance of the feature to be removed
        beta = ridge_resression(X_temp, y, lambda_)
        y_hat = predict(X_temp, beta)

        var = variance_least_squares_weights(X_temp, y, y_hat)

        ttest_result = ttest(np.shape(X_temp), beta[indices.index(index_to_delete)],
                             var[indices.index(index_to_delete)], tolerance=0.95)

        # if the feature is irrelevant, remove it from indices
        if not ttest_result:
            indices.remove(index_to_delete)

    return indices


def simple_to_general(X: np.ndarray, y: np.ndarray) -> List[int]:
    """
    Finds the relevant features using the simple to general approach.

    :param X: The matrix of observables
    :param y: The outcome matrix
    :return: list of indices
    """

    n, k = np.shape(X)

    # indices encloses the features to use
    # remaining_indices encloses the features that are available but not in indices
    indices = []
    remaining_indices = list(range(k))

    # ttest_result encloses the relevance of the feature in question
    ttest_result = True

    # while the feature in question is relevant and there are less than k features
    while ttest_result and len(indices) < k:

        # initialize the feature to be added and its r squared
        index_to_add = remaining_indices[0]
        r_2 = -math.inf

        # find the feature that yields the largest r squared if added
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

        # supposing it's relevant, add the feature
        indices.append(index_to_add)

        # the first feature is always added, we only test relevance if it's 2nd or more
        if len(indices) > 1:
            # keep only the features in indices
            X_temp = np.copy(X[:, indices])

            # compute the relevance of the feature
            beta = least_squares(X_temp, y)
            y_hat = predict(X_temp, beta)

            var = variance_least_squares_weights(X_temp, y, y_hat)

            ttest_result = ttest(np.shape(X_temp), beta[indices.index(index_to_delete)],
                                 var[indices.index(index_to_delete)], tolerance=0.95)

            # if the feature turn out to be irrelevant, remove it
            if not ttest_result:
                indices.remove(index_to_add)

    return indices
