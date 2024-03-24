import numpy as np

def loss(y_pred, Y):
    nll = -(np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred)) / Y.shape[1])

    return nll

def accuracy(y_pred, y):

  return np.sum(y_pred == y) / y.shape[1] * 100