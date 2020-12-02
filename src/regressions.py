import numpy as np


def regression(dataframe):
   "normal linear regression"
   # I delete all na of the data set, find the Real Gross Domestic somewhere and assume it always as our y
   # I return the beta then

   
   dataframe=dataframe.dropna();
   
   y = dataframe.filter(regex='REAL GROSS DOMESTIC');
   y = y.values
   
   x = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='REAL GROSS DOMESTIC')))];
   x = x.values
    
   oner = np.ones((len(x),1))
   x=np.hstack((oner,x))


   beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x)),y)
   
   
   return beta

