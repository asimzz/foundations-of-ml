import numpy as np
from activations import *

class SingleLayerNN:
    def __init__(self, h0, h1, h2):
        self.W1 = np.random.rand(h1,h0)
        self.W2 = np.random.rand(h2,h1)
        self.b1 = np.random.rand(h1,1)
        self.b2 = np.random.rand(1,1)
        
    def forward_pass(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = sigmoid(Z2)
        return A2, Z2, A1, Z1
    
    def backward_pass(self, X, Y, A2, A1, Z1):
        size = Y.shape[1]
        dL_dZ2 = A2 - Y
        dZ2_dW2 = A1.T
        dW2 = np.dot(dL_dZ2, dZ2_dW2) / size

        db2 = (np.sum(dL_dZ2, axis =1, keepdims=True))/ size

        dZ1_dW1 = X.T
        dL_dZ1 = (self.W2.T @ dL_dZ2) * d_sigmoid(Z1)
        dW1 = np.dot(dL_dZ1, dZ1_dW1) / size
        db1 = (np.sum(dL_dZ1, axis =1, keepdims=True))/ size
        
        return dW1, dW2, db1, db2
    
    def update(self, dW1, dW2, db1, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2
        
    def predict(self, X):
        A2, _, _, _ = self.forward_pass(X)
        predictions = (A2 >= 0.5).astype(int)
        return predictions