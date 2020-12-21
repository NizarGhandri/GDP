import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.regressions import ridge_regression
from src.evaluation_metrics import R_squared
import matplotlib.pyplot as plt


def X_y_from_dataset(dataset: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Clean dataset (drop Nan) and separate features and labels.

    :param dataset: data dataframe
    :return: separated features and labels
    """
    df = dataset.dropna()
    y = df.filter(regex='REAL GROSS DOMESTIC')
    return df.drop(columns=y.columns).values, y.values


def z_score_scaling(X: np.ndarray) -> np.ndarray:
    """
    Normalize the dataset using mean in standard deviation to make expected value 0 and variance 1.

    :param X: features
    :return: normalized features
    """
    return (X - np.mean(X)) / np.std(X)


def min_max_scaling(X: np.ndarray) -> np.ndarray:
    """
    Standardize the dataset using min max scaling.

    :param X: features
    :return: standardized features
    """
    min_X = np.min(X, axis=0)
    return (X - min_X) / (np.max(X, axis=0) - min_X)


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Predict results of linear regression given the weights and the feature matrix.

    :param X: features
    :param w: weights
    :return: prediction
    """
    return X @ w


def add_bias(X: np.ndarray, b: float = 1) -> np.ndarray:
    """
    Concatenate a bias to the the dataset.

    :param X: features
    :param b: bias
    :return: augmented dataset
    """
    return np.hstack([b * np.ones((len(X), 1)), X])


def train_test_split(X, y, proportion=0.8) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Split dataset into training and test using the Year feature and a rate of 80%.

    :param X: features
    :param y: labels
    :param proportion: proportion of train from full dataset
    :return: train and test features and labels
    """
    n = int(len(y) * proportion)
    return X[:n], X[n:], y[:n], y[n:]


def test_ridge_reg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                   lambda_: float) -> float:
    """
    Compute testing R squared for ridge regression fittet on training set with lambda_ as a penalty term

    :param X_train: training features
    :param y_train: training labels
    :param X_test: testing features
    :param y_test: testing labels
    :param lambda_: penalty term
    :return: r_squared
    """
    return R_squared(y_test, predict(X_test, ridge_regression(X_train, y_train, lambda_)))


def cross_val_ridge(X: np.ndarray, y: np.ndarray, min_lambda: float = 0, max_lambda: float = 0,
                    step: int = 0) -> float:
    """
    Find best lambda using cross validation

    :param X: featurs
    :param y: labels
    :param min_lambda: minimum lambda
    :param max_lambda:  maximum lambda
    :param step: step between lambdas
    :return: best lambda value
    """
    kf = KFold(n_splits=5, shuffle=True)
    lambdas = np.linspace(min_lambda, max_lambda, step)

    r_2 = []
    for lambda_ in lambdas:
        mean = np.mean([test_ridge_reg(X[train_index], y[train_index], X[test_index], y[test_index], lambda_)
                        for train_index, test_index in kf.split(X)])
        r_2.append(mean)

    best_lambda_index = np.argmax(r_2)[0]
    best_lambda = lambdas[best_lambda_index]

    plt.plot(lambdas, r_2)
    plt.xlabel("lambda")
    plt.ylabel("R_squared")
    plt.scatter(best_lambda, r_2[best_lambda_index], color='red')

    return best_lambda
