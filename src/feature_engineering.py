import numpy as np


def dummy_variable(x, predicate, column_list):
    """
    adding a dummy variable to the dataset
    """
    return np.hstack(predicate(*(x[:,column_list].T)), x)


def build_poly(x, degree): 
    """
    polynomial expansion
    """
    expanded = np.ones((x.shape[0],1))
    for idx in range(1,degree+1): expanded = np.hstack((expanded, x**idx))
    return expanded


def build_log(x):
    """
    logarithmic expansion 
    """
    expanded = np.ones((x.shape[0],1))
    expanded = np.hstack((expanded, np.nan_to_num(np.log(x))))
    return expanded


def build_trigo(x):
    """
    trigonometric expansion
    """
    expanded = np.ones((x.shape[0],1))
    expanded = np.hstack((np.hstack((expanded, np.cos(x))),np.sin(x)))
    return expanded


def build_hyperbolic(x):
    """
    hyperbolic expansion
    """
    expanded = np.ones((x.shape[0],1))
    expanded = np.hstack((np.hstack((expanded, np.sinh(x))),np.cosh(x)))
    return expanded

