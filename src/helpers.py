import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.regressions import ridge_regression, least_squares
from src.evaluation_metrics import R_squared
import matplotlib.pyplot as plt
from src.feature_engineering import build_poly
from mpl_toolkits import mplot3d


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


def train_test_split(X, y, proportion=0.8, shuffle=False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Split dataset into training and test using the Year feature and a rate of 80%.

    :param X: features
    :param y: labels
    :param proportion: proportion of train from full dataset
    :return: train and test features and labels
    """
    n = int(len(y) * proportion)

    if shuffle:
        rng = np.random.default_rng()
        indices = np.arange(len(y))
        rng.shuffle(indices)
        train = indices[:n]
        test = indices[n:]
        return X[train], X[test], y[train], y[test]
    else:
        return X[:n], X[n:], y[:n], y[n:]


def _test_ridge_reg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
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


def degree_cross_val(X: np.ndarray, y: np.ndarray, max_degree: int, plot: bool = True):
    """
    Find best degree for a polynomial expansion using K-fold cross validation

    :param X: features
    :param y: labels
    :param max_degree:  maximum lambda
    :return: best degree value
    """
    kf = KFold(n_splits=5, shuffle=True)
    degrees = np.arange(0, max_degree+1)

    r_2 = []
    for degree in degrees:
        X_expanded = build_poly(X, degree)
        mean = np.mean([_test_ridge_reg(X_expanded[train_index], y[train_index], X_expanded[test_index], y[test_index], 0)
                        for train_index, test_index in kf.split(X_expanded)])
        r_2.append(mean)

    best_degree_index = np.argmax(r_2)
    
    best_degree = degrees[best_degree_index]
    
    if plot: 
        plt.plot(degrees, r_2)
        plt.xlabel("degree")
        plt.ylabel("R_squared")
        plt.scatter(best_degree, r_2[best_degree_index], color='red')

    return best_degree
                  


def cross_val_ridge(X: np.ndarray, y: np.ndarray, plot: bool = True, min_lambda: float = 0, max_lambda: float = 1, max_degree: int = 10,
                    n_points: int = 100) -> float:
    """
    Find best lambda using K-fold cross validation

    :param X: features
    :param y: labels
    :param min_lambda: minimum lambda
    :param max_lambda:  maximum lambda
    :param n_points: number of lambdas to test
    :return: best pair (lambda, degree) value
    """
    
    kf = KFold(n_splits=5, shuffle=True)
    lambdas = np.linspace(min_lambda, max_lambda, n_points)

    r_2 = []
    for lambda_ in lambdas:
        for degree in range(1, max_degree+1): 
            X_expanded = build_poly(X, degree)
            mean = np.mean([_test_ridge_reg(X_expanded[train_index], y[train_index], X_expanded[test_index], y[test_index], lambda_)
                            for train_index, test_index in kf.split(X_expanded)])
            r_2.append(mean)
            
    argmax = np.argmax(r_2)
    best_lambda_index = argmax//max_degree
    best_degree = argmax%max_degree + 1
    best_lambda = lambdas[best_lambda_index]
    
    if plot: 
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(np.array(lambdas), np.arange(1, max_degree+1), np.array(r_2), rstride=1, cstride=1, cmap='viridis', edgecolor='none');
        #plt.plot(lambdas, r_2)
        ax.set_xlabel("lambda")
        ax.set_ylabel("degree")
        ax.set_zlabel("R_squared")
        #plt.scatter(best_lambda, r_2[best_lambda_index], color='red')

    return best_lambda, best_degree
