import numpy as np
from scipy.stats import t
from scipy import stats
from itertools import chain, combinations
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2
from numpy import linalg as LA
from src.evaluation_metrics import R_squared
from src.regressions import least_squares
from src.helpers import train_test_split, predict
import copy


def confidence_interval(x_shape, variable, var, tolerance=0.95):
    """
    calculating confidence interval
    :param x_shape:
    :param variable:
    :param var:
    :param tolerance:
    :return:
    """
    # giving back a n x 2 matrix, on the first column lower bound, second column upper bound
    # x - explanatory variables
    # beta - parameteres
    # var - calculated variance (can differ dependent on model)
    # tolerance - which tolerance level do you want

    n, m = x_shape
    degoffree = n - m
    alpha = stats.t.ppf(1-(1 - tolerance) / 2, degoffree) * np.sqrt(var)
    return variable - alpha, variable + alpha


def standard_error_regression(y, y_hat, X_shape):
    """
    calculating Standard Error
    :param y:
    :param y_hat:
    :param X_shape:
    :return:
    """
    e = y - y_hat
    return (e.T @ e) / (X_shape[0] - X_shape[1])


def variance_least_squares_weights(y, y_hat, X):
    """
    calculating variance of least squares
    :param y:
    :param y_hat:
    :param X:
    :return:
    """
    return standard_error_regression(y, y_hat, X.shape) * np.reshape(np.diag(np.linalg.inv(X.T @ X)), (-1, 1))


# def variance_least_squares_line (y, y_hat, X):
#     var = (X - np.mean(X, axis=0))**2
#     return (var/sum(var) + 1/X.shape[0]) * standard_error_regression (y, y_hat, X.shape)


def variance_least_squares_line(y, y_hat, X):
    """
    calculating variance of least squares
    :param y:
    :param y_hat:
    :param X:
    :return:
    """
    return standard_error_regression(y, y_hat, X.shape) * (
        np.reshape(np.diag(X @ np.linalg.inv(X.T @ X) @ X.T), (-1, 1)))


def subset_iterator(X_columns):
    """
    To use this to find a all the subsets of X you do the following: 
    for columns in subset_iterator(range(X.shape[1])):
        X[:, columns] # this will be your new dataset out of the subsets 

    :param X_columns:
    :return:
    """
    rnge = range(X_columns)
    return chain(*map(lambda x: combinations(rnge, x), range(2, X_columns + 1)))



def best_subset(X, y): 
    scores = []
    subsets = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)
    for i in subset_iterator(X.shape[1]):
        ws = least_squares(X_train[:, i], y_train)
        scores.append(R_squared(y_test, predict(X_test[:, i], ws)))
        subsets.append(i)
        
    return scores, subsets




def ttest(x_shape, betak, vark, tolerance=0.95):
    """
    Computes the statistical significance of a specific variable
    :x_shape: shape of the observed matrix
    :betak: estimator of the specific parameter
    :vark: variance of specific parameter
    
    :return: true if it is statistically significant, false if it is not
    """

    n,m=x_shape
    degoffree=n-m
    tt=stats.t.ppf(1-(1-tolerance)/2,degoffree)
    tk=betak/np.sqrt(vark)
    if tk>tt:
        test=True    
    else:
        test=False
        
    return test
    

    
def breusch_pagan_test(x, y):
    '''
    taken from:
    https://stackoverflow.com/questions/30061054/ols-breusch-pagan-test-in-python
    Breusch-Pagan test for heteroskedasticity in a linear regression model:
    H_0 = No heteroskedasticity.
    H_1 = Heteroskedasticity is present.

    Inputs:
    x = a numpy.ndarray containing the predictor variables. Shape = (nSamples, nPredictors).
    y = a 1D numpy.ndarray containing the response variable. Shape = (nSamples, ).

    Outputs a list containing three elements:
    1. the Breusch-Pagan test statistic.
    2. the p-value for the test.
    3. the test result.
    '''
    y = y.ravel()
    if y.ndim != 1:
        raise SystemExit('Error: y has more than 1 dimension.')
    if x.shape[0] != y.shape[0]:
        raise SystemExit('Error: the number of samples differs between x and y.')
    else:
        n_samples = y.shape[0]

    # fit an OLS linear model to y using x:
    lm = LinearRegression()
    lm.fit(x, y)

    # calculate the squared errors:
    err = (y - lm.predict(x))**2

    # fit an auxiliary regression to the squared errors:
    # why?: to estimate the variance in err explained by x
    lm.fit(x, err)
    pred_err = lm.predict(x)
    del lm

    # calculate the coefficient of determination:
    ss_tot = sum((err - np.mean(err))**2)
    ss_res = sum((err - pred_err)**2)
    r2 = 1 - (ss_res / ss_tot)
    del err, pred_err, ss_res, ss_tot

    # calculate the Lagrange multiplier:
    LM = n_samples * r2
    del r2

    # calculate p-value. degrees of freedom = number of predictors.
    # this is equivalent to (p - 1) parameter restrictions in Wikipedia entry.
    pval = chi2.sf (LM, x.shape[1])

    if pval < 0.05:
        test_result = 'Heteroskedasticity present at 95% CI.'
    else:
        test_result = 'No significant heteroskedasticity.'
    return [LM, pval, test_result]

def condition_number(x):
    """
    calculates the Condition Number, the bigger the worse the multicolinearity, starts to become a problem from 20 on
    :param x: Observed matrix
    :return: condition number
    """
    w, v = LA.eig(np.dot(np.transpose(x),x))
    return np.sqrt(np.max(w)/np.min(w))


def VIF(x):
    """
    calculates the Variance Inflation Factor, the bigger the worse the multicolinearity
    :param x: Observed matrix
    :return: VIF
    """
    
    shape=x.shape
    shape=shape[1]
    xtemp2 = copy.copy(x)
    VIFF=np.zeros(shape)
    
    for i in range(0, shape):
        if i==0:
            x0=xtemp2[:,1:]
            y0=xtemp2[:,0]
        elif i==shape:
            x0=xtemp2[:,0:-1]
            y0=xtemp2[:,shape]
        else:
            x1=xtemp2[:,:i]
            x2=xtemp2[:,i+1:]
            x0=np.hstack((x1,x2))
            y0=xtemp2[:,i]
    
        beta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x0),x0)),np.transpose(x0)),y0)
        y_hat=np.dot(x0,beta)
        VIFF[i]=1/(1-R_squared(y0, y_hat))
        
    return VIFF



def general_to_simple(x,y,tolerance=0.95):
    """
    Compute the model from the general to simple approach

    :param X: The matrix of observables
    :param y: The outcome matrix
    :return: model from the general to simple approach
    """
    shape=np.shape(x)
    xtemp2 = copy.copy(x)
    a=np.zeros(shape[1])

    for f in range(shape[1]):
        
        for i in range(shape[1]-f):
            
            if i==0:
                x0=xtemp2[:,1:]

            elif i==(shape[1]-f):
                x0=xtemp2[:,0:-1]

            else:
                x1=xtemp2[:,:i]
                x2=xtemp2[:,i+1:]
                x0=np.hstack((x1,x2))

            beta_reduced = least_squares(x0, y)
            y_hat_reduced=np.dot(x0,beta_reduced)
            a[i] = R_squared(y, y_hat_reduced)

        ind = np.argmax(a[:])
        beta_full = least_squares(xtemp2, y)
        y_hat_full=np.dot(xtemp2,beta_full)
        var = variance_least_squares_line(y, y_hat_full, xtemp2)
        shapex2=np.shape(xtemp2)
        error=np.dot(np.transpose(y-np.dot(xtemp2,beta_full)),y-np.dot(xtemp2,beta_full))/(shapex2[0]-shapex2[1])
        var=np.linalg.inv(np.dot(np.transpose(xtemp2),xtemp2))*error
        stat_sign=ttest(np.shape(xtemp2), beta_full[ind], var[ind,ind], tolerance=0.95)

        if stat_sign:
            return xtemp2
        else:
            xtemp2 = np.delete(xtemp2, ind, axis=1)
            a=np.zeros(shape[1]-f)
            
        del x0, x1, x2, beta_reduced, y_hat_reduced, ind, beta_full, y_hat_full, var, stat_sign

    
    return xtemp2



def simple_to_general(x,y,tolerance=0.95):
    """
    Compute the model from the simple to general approach
    :param X: The matrix of observables
    :param y: The outcome matrix
    
    :return: model from the simple to general approach
    """
    
    shape=np.shape(x)
    xtemp2 = copy.copy(x)
    a=np.zeros(shape[1])

    for f in range(shape[1]-1):
        for i in range(shape[1]-f):
            if f==0:
                x0=xtemp2[:,i]
                x0=x0.transpose()
                x0=np.expand_dims(x0, axis=1)
                beta_reduced = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x0),x0)),np.transpose(x0)),y)
                y_hat_reduced=np.dot(x0,beta_reduced)
                a[i] = R_squared(y, y_hat_reduced)

            else:
                x1=xtemp2[:,i]
                x1=np.expand_dims(x1, axis=1)
                x2=np.hstack((x0,x1))
                beta_reduced = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x2),x2)),np.transpose(x2)),y)
                y_hat_reduced=np.dot(x2,beta_reduced)
                a[i] = R_squared(y, y_hat_reduced)                  
        if f==0:
            stat_sign=False
            ind = np.argmax(a[:])
            x0=xtemp2[:,ind]
            x0=np.expand_dims(x0, axis=1)

        else:
            ind = np.argmax(a[:])
            x1=xtemp2[:,ind]
            x1=np.expand_dims(x1, axis=1)
            x2=np.hstack((x0,x1))
            beta_full = least_squares(x2, y)
            y_hat_full=np.dot(x2,beta_full)
            shapex2=np.shape(x2)
            error=np.dot(np.transpose(y-np.dot(x2,beta_full)),y-np.dot(x2,beta_full))/(shapex2[0]-shapex2[1])
            var=np.linalg.inv(np.dot(np.transpose(x2),x2))*error
            stat_sign=ttest(np.shape(x2), beta_full[f,0], var[f,f], tolerance=0.95)    
            del x1, beta_full,y_hat_full,var, error, shapex2

        if stat_sign:
            return x2
        else:
            xtemp2 = np.delete(xtemp2, ind, axis=1)
            if f==0:
                r=1
                a=np.zeros(shape[1])
            else:
                x0=x2
                a=np.zeros(shape[1])

    return x0
