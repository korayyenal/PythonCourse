import pandas as pd
import numpy as np
from scipy import stats

def linearregression():
  dfmerge = pd.read_csv('dfmerge.csv')
  x=dfmerge[['Scientific_And_Technical_Journal_Articles','Literacy_Rate']]
  y=dfmerge['GDPpc']
  
  y= np.log(y)
  
  xones = np.ones((123,1), int)
  
  x1 = np.array(dfmergecopy["Scientific_And_Technical_Journal_Articles"])
  x2 = np.array(dfmergecopy["Literacy_Rate"])
  
  xfin=np.hstack([x1[:, np.newaxis], x2[:, np.newaxis]])
  scaledXfin=(xfin-xfin.mean(axis=0))/xfin.std(0)
  X=np.hstack([xones, scaledXfin])
  
  yfin=y[:, np.newaxis]
  
  xTx = X.T @ X
  xTy = X.T @ yfin
  
  B = np.linalg.inv(xTx) @ xTy
  
  Bo=B[0].item(0)
  B1=B[1].item(0)
  B2=B[2].item(0)
  
  print("Regression coefficients are Bo =",Bo,", B1 =",B1,", B2 =",B2)
  print("Fitted regression model is y=",Bo,"+",B1,"x1 +",B2,"x2")
  
  yline=[]
  
  for i in range(1,370, 3):
        xo1=X.item(i)
        xo2=X.item(i+1)
        yt=Bo+xo1*B1+xo2*B2
        yline.append(yt)
        
  ylinefin=np.asarray(yline)
  yprime=ylinefin[:, np.newaxis]
  
  
  e=yfin-yprime
  print("e=y-yi values: ")
  print(e)
  
  n = 123 #sample size
  k = 3 #number of variables
  sigmasq=(e.T @ e)/(n-k-1)
  
  print("Sigma_sq:" ,sigmasq)
  stderror=sigmasq*np.linalg.inv(xTx)
  
  #finding the variance of Beta
  print("Var(B):", stderror)
  
  #finding the standard error
  stdcoeferrs=[stderror.item(0), stderror.item(4), stderror.item(8)]
  stdcoefi=np.asarray(stdcoeferrs)
  stdcoefff=np.sqrt(stdcoeff)
  stdcoeff=stdcoefi[:, np.newaxis]
  
  print("Standard errors:")
  print(stdcoefff)
  
  #finding the t value of 95% CI with n & k values
  t = stats.t.ppf(.975,n-k-1)
  #defining lower and upper bounds
  l1=Bo-stdcoefff[0]*t
  l2=B1-stdcoefff[1]*t
  l3=B2-stdcoefff[2]*t
  u1=Bo+stdcoefff[0]*t
  u2=B1+stdcoefff[1]*t
  u3=B2+stdcoefff[2]*t
  
  lower=[l1,l2,l3]
  upper=[u1,u2,u3]
  
  lowerfinal=np.asarray(lower)
  upperfinal=np.asarray(upper)
  
  confidence=np.append(lowerfinal, upperfinal, axis=1)
  
  print("Lower interval:")
  print(lowerfinal)
  print("Upper interval:")
  print(upperfinal)
  
  table=np.append(B, stdcoefff, axis=1)
  table=np.append(table, confidence, axis=1)
  
  summarytable = pd.DataFrame(table,index=['Bo','B1','B2'],columns=['Estimates', 'Standard Errors', 'Lower CI', 'Upper CI'])
  print(summarytable)
  
