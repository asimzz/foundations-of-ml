import numpy as np


def sigmoid(z):

  return 1/(1 + np.exp(-z))

# sigmoid first derivative
def d_sigmoid(z):

  a = sigmoid(z)

  return a * (1 - a)