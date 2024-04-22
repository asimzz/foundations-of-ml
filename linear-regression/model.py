import numpy as np
from utils import mean_squared_error

class LinearRegression:
    def __init__(self):
        self.theta = None
        
    def forward_pass(self, X):
        return np.dot(X, self.theta)
        
    def backward_pass(self, X, y):
        yhat = self.forward_pass(X)
        dtheta =  -2 * np.dot(X.T, (y - yhat))
        return dtheta
    
    def update_param(self, grads, step_size=0.1):
        self.theta -= step_size * grads
        
    def fit(self, Xtrain, ytrain, n_epochs=10):
        _, self.D = Xtrain.shape
        self.theta = np.zeros((self.D,1))
        losses = []
    
        for epoch in range(n_epochs):
            ypred = self.forward_pass(Xtrain)
            loss = mean_squared_error(ytrain, ypred)
            grads = self.backward_pass(Xtrain, ytrain)
            self.update_param(grads)

            print(f"\nEpoch {epoch}, loss {loss}")

        
