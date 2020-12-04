import numpy as np

def X_y_from_dataset (dataset): 
    """
    Clean dataset (drop Nan) and separate it in X for predictor columns and y for for the real gross domestic
    """
    df = dataset.dropna()
    y = df.filter(regex='REAL GROSS DOMESTIC')
    return df.drop(columns=y.columns).values, y.values 


def z_score_scaling (X):
    """
    Normalize the dataset using mean in standard deviation to make expected value 0 and variance 1
    """
    return (X - np.mean(X))/np.std(X)

def min_max_scaling (X):
    """
    Standardize the dataset using min max scaling
    """ 
    min_X = np.min(X, axis=0)
    return (X - min_X)/(np.max(X, axis=0) - min_X)


def predict(X, w): 
    """
    Predicts results of linear regression given the weights and the feature matrix
    """
    return X@w

def add_bias(X, b=1): 
    """
    Concatenates a bias to the the dataset
    """
    return np.hstack([np.ones(len(X), 1), X])


def split(X, y, rate=0.8):
    """
    Splits dataset into training and test using the Year feature and a rate f 80%
    """
    n = int(len(y)*rate)
    return X[:n], X[n:], y[:n], y[n:]




