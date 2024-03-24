import numpy as np

def mean(X): # np.mean(X, axis = 0)
  mean = np.sum(X, axis=0)/len(X)

  return mean

def std(X): # np.std(X, axis = 0)
  squared_diff = (X-mean(X))**2
  varaiance = np.sum(squared_diff,axis=0)/(len(X)-1)
  std = np.sqrt(varaiance)

  return std

def standardize_data(X):
  X_std = (X - mean(X))/std(X)

  return X_std

def covariance(X):
  cov = X.T @ X/(len(X)-1)

  return cov