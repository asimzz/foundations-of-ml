import numpy as np
from utils import cross_entropy
class LogisticRegression:
  '''
  The goal of this class is to create a LogisticRegression class,
  that we will use as our model to classify data point into a corresponding class
  '''
  def __init__(self):
    self.w = None

  def add_ones(self, x):
    ones = np.ones((x.shape[0],1))
    return np.hstack((ones,x))

  def sigmoid(self, x):
    z = x @ self.w
    sig = 1 / (1+np.exp(-z))
    return sig


  def predict_proba(self,x):  
    x = self.add_ones(x)
    proba = self.sigmoid(x)
    return proba

  def predict(self,x):
    probas = self.predict_proba(x)
    #convert the probalities into 0 and 1 by using a treshold=0.5
    output = np.where(probas >= 0.5, 1, 0) 
    return output

  def fit(self,x,y, lr, n_epochs=10000):

    x = self.add_ones(x)

    y = y.reshape(-1,1)

    self.w = np.zeros((x.shape[1],1))

    for epoch in range(n_epochs):
      # make predictions
      y_pred = self.sigmoid(x)

      N = x.shape[0]
      grads = (- 1/N) * x.T @ (y - y_pred)

      self.w -= lr * grads

      loss = cross_entropy(y, y_pred)

      if epoch%1000 == 0:
        print(f'Loss for epoch {epoch}  : {loss}')
