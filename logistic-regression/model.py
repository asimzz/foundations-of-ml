import numpy as np

class LogisticRegression:
  '''
  The goal of this class is to create a LogisticRegression class,
  that we will use as our model to classify data point into a corresponding class
  '''
  def __init__(self):
    self.train_losses = []
    self.w = None

  def add_ones(self, x):
    ones = np.ones((x.shape[0],1))
    return np.hstack((ones,x))

  def sigmoid(self, x):
    z = x @ self.w
    sig = 1 / (1+np.exp(-z))
    return sig

  def cross_entropy(self, x, y_true):
    N = x.shape[0]
    y_pred = self.sigmoid(x)
    loss = - np.mean(y_true * np.log(y_pred) + (1-y_true)* np.log(1-y_pred))
    return loss


  def predict_proba(self,x):  #This function will use the sigmoid function to compute the probalities
    x = self.add_ones(x)
    proba = self.sigmoid(x)
    return proba

  def predict(self,x):
    probas = self.predict_proba(x)
    output = np.where(probas >= 0.5, 1, 0) #convert the probalities into 0 and 1 by using a treshold=0.5
    return output

  def fit(self,x,y, lr, n_epochs=10000):

    # Add ones to x
    x = self.add_ones(x)

    # reshape y if needed
    y = y.reshape(-1,1)

    # Initialize w to zeros vector >>> (x.shape[1])
    self.w = np.zeros((x.shape[1],1))

    for epoch in range(n_epochs):
      # make predictions
      preds = self.sigmoid(x)

      #compute the gradient
      N = x.shape[0]
      grads = (- 1/N) * x.T @ (y - preds)

      #update rule
      self.w -= lr * grads

      #Compute and append the training loss in a list
      loss = self.cross_entropy(x, y)
      self.train_losses.append(loss)

      if epoch%1000 == 0:
        print(f'Loss for epoch {epoch}  : {loss}')

  def check_accuracy(self,y_true, y_pred):
    acc = np.mean(y_true.reshape(-1,1)==y_pred) * 100
    return acc