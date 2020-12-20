import numpy as np
from sklearn.model_selection import KFold
from src.regressions import ridge_regression
from src.evaluation_metrics import R_squared


def X_y_from_dataset(dataset):
    """
    Clean dataset (drop Nan) and separate it in X for predictor columns and y for for the real gross domestic
    :param dataset:
    :return:
    """
    df = dataset.dropna()
    y = df.filter(regex='REAL GROSS DOMESTIC')
    return df.drop(columns=y.columns).values, y.values


def z_score_scaling(X):
    """
    Normalize the dataset using mean in standard deviation to make expected value 0 and variance 1
    :param X:
    :return:
    """
    return (X - np.mean(X)) / np.std(X)


def min_max_scaling(X):
    """
    Standardize the dataset using min max scaling
    :param X:
    :return:
    """
    min_X = np.min(X, axis=0)
    return (X - min_X) / (np.max(X, axis=0) - min_X)


def predict(X, w):
    """
    Predicts results of linear regression given the weights and the feature matrix
    :param X:
    :param w:
    :return:
    """
    return X @ w


def add_bias(X, b=1):
    """
    Concatenates a bias to the the dataset
    :param X:
    :param b:
    :return:
    """
    return np.hstack([b * np.ones((len(X), 1)), X])


def split(X, y, rate=0.8):
    """
    Splits dataset into training and test using the Year feature and a rate of 80%
    :param X:
    :param y:
    :param rate:
    :return:
    """
    n = int(len(y) * rate)
    return X[:n], X[n:], y[:n], y[n:]


def test_ridge(X_train, y_train, X_test, y_test, lambda_):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param lambda_:
    :return:
    """
    return R_squared(y_test, predict(X_test, ridge_regression(X_train, y_train, lambda_)))


def cross_val_ridge(X, y, linear_space):
    """

    :param X:
    :param y:
    :param linear_space:
    :return:
    """
    kf = KFold(n_splits=5, shuffle=True)
    results = []
    for lambda_ in linear_space:
        mean = np.mean([test_ridge(X[train_index], y[train_index], X[test_index], y[test_index], lambda_) \
                        for train_index, test_index in kf.split(X)])
        results.append(mean)
    return np.argmax(results), results
