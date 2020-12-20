import numpy as np
from typing import Callable


def dummy_variable(x: np.ndarray, predicate: Callable[[float], bool], column_list: list) -> np.ndarray:
    """
    Add a dummy variable to the dataset.

    :param x: features
    :param predicate: a predicate function
    :param column_list: list of columns to which the predicate is applied
    :return: features augmented wit dummy variables
    """
    return np.hstack(predicate(*x[:, column_list].T), x)


def build_poly(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Polynomial expansion.

    :param x: features
    :param degree: degree
    :return: augmented features
    """
    expanded = np.ones((x.shape[0], 1))
    for idx in range(1, degree + 1): expanded = np.hstack((expanded, x ** idx))
    return expanded


def build_log(x: np.ndarray) -> np.ndarray:
    """
    Logarithmic expansion.

    :param x: features
    :return: augmented features
    """
    expanded = np.ones((x.shape[0], 1))
    expanded = np.hstack((expanded, np.nan_to_num(np.log(x))))
    return expanded


def build_trigo(x: np.ndarray) -> np.ndarray:
    """
    Trigonometric expansion usng sin and cos.

    :param x: features
    :return: augmented features
    """
    expanded = np.ones((x.shape[0], 1))
    expanded = np.hstack((np.hstack((expanded, np.cos(x))), np.sin(x)))
    return expanded


def build_hyperbolic(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic expansion using sinh and cosh.

    :param x: features
    :return: augmented features
    """
    expanded = np.ones((x.shape[0], 1))
    expanded = np.hstack((np.hstack((expanded, np.sinh(x))), np.cosh(x)))
    return expanded
