import numpy as np

# Assuming X and y are a columns vectors with shapes m x 1 and n x 1 respectively
def get_distance(X, y):
    x_size = X.shape[0]
    y_size = y.shape[0]
    dist = np.zeros((x_size, y_size))
    dist = np.sqrt(np.sum(np.square(X),axis=1).reshape(-1,1) + np.sum(np.square(y),axis=1) - 2 * np.dot(X,y.T))
    return dist

def check_accuracy(y_pred, y, size):

  result = np.sum(y_pred == y) / size * 100
  return round(result, 2)