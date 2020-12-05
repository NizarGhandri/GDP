import numpy as np


def least_squares(X, y):
   """
   Compute least squares with np.solve for more numerical stability 
   """
   X_t = X.T
   return np.linalg.solve(X_t.dot(X), X_t.dot(y))



