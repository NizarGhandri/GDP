import numpy as np


def dummy_variable(x, predicate, column_list):
    """
    adding a dummy variable to the dataset
    :param x:
    :param predicate:
    :param column_list:
    :return:
    """
    return np.hstack(predicate(*(x[:, column_list].T)), x)


def build_poly(x, degree):
    """
    polynomial expansion
    :param x:
    :param degree:
    :return:
    """
    expanded = np.ones((x.shape[0], 1))
    for idx in range(1, degree + 1): expanded = np.hstack((expanded, x ** idx))
    return expanded


def build_log(x):
    """
    Logarithmic expansion.

    :param x: features
    :return: augmented features.
    """
    expanded = np.ones((x.shape[0], 1))
    expanded = np.hstack((expanded, np.nan_to_num(np.log(x))))
    return expanded


def build_trigo(x):
    """
    Trigonometric expansion usng sin and cos.

    :param x: features
    :return: augmented features.
    """
    expanded = np.ones((x.shape[0], 1))
    expanded = np.hstack((np.hstack((expanded, np.cos(x))), np.sin(x)))
    return expanded


def build_hyperbolic(x):
    """
    Hyperbolic expansion using sinh and cosh.

    :param x: features
    :return: augmented features.
    """
    expanded = np.ones((x.shape[0], 1))
    expanded = np.hstack((np.hstack((expanded, np.sinh(x))), np.cosh(x)))
    return expanded
