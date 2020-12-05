import numpy as np
from scipy.stats import t
from scipy import stats


def correlation_test(X, threshold=0.95):
    correlation_matrix = np.corrcoef(X)
    correlation_matrix < threshold 
    return correlation_matrix


def confidenceinterval(x, beta, var, tolerance=1-0.95):
   """
   calculating confidence interval
   """
   # giving back a n x 2 matrix, on the first column lower bound, second column upper bound
   # x - explanatory variables
   # beta - parameteres
   # var - calculated variance (can differ dependent on model)
   # tolerance - which tolerance level do you want
    
   degoffree=len(x)-len(x[0])
   tstatvalue=stats.t.ppf(1-tolerance, degoffree)
   
   interval = np.zeros((len(x[0]),2))
    
   interval[:,0]=np.squeeze(beta-tstatvalue*np.sqrt(var))

   interval[:,1]=np.squeeze(beta+tstatvalue*np.sqrt(var))
   
   
   
   return interval