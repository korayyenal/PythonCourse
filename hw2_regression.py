import sys
import pandas as pd
import numpy as np
from scipy import stats
import readdata

def linearregression(X,y):
  
  if X is None:
        raise TypeError("X must be provided")
    if y is None:
        raise TypeError("y must be provided")

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be numpy ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be numpy ndarray")

   n, k = X.shape
   q, p = y.shape
  
   X = np.hstack([np.ones((n,1)), X])
    
   B = np.linalg.inv(X.T @ X) @ X.T @ y
   Bo=B[0].item(0)
   B1=B[1].item(0)
   B2=B[2].item(0)
   print("Regression coefficients are Bo =",Bo,", B1 =",B1,", B2 =",B2)
   print("Fitted regression model is y=",Bo,"+",B1,"x1 +",B2,"x2")
  
   y_est = X @ B
   residual = y - y_est
   print("e=y-yi values: ")
   print(e)
  
   sigmasq = residual.T @ residual / (n - k - 1)
   print("Sigma_sq:" ,sigmasq)
  
   var = np.diag( np.multiply(sigmasq, np.linalg.inv(X.T @ X)))
   print("Var(B):", stderror)
  
   standard_err = np.sqrt(var).reshape(var.size,1)
   print("Standard errors:")
   print(standard_err)
    
   CIC  = 0.05
   t_stat = stats.t.ppf(1 - CIC/2, n-k-1)
   q_err = t_stat * standard_err
   conf = np.hstack([B - q_err, B + q_err])
    
   table=np.append(B, stdcoefff, axis=1)
   table=np.append(table, confidence, axis=1)
    
   summarytable = pd.DataFrame(table,index=['Bo','B1','B2'],columns=['Estimates', 'Standard Errors', 'Lower CI', 'Upper CI'])
   print(summarytable)
  
   return B, standard_err, conf
  

  

  
  
