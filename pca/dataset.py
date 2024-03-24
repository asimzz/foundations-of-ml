import pandas as pd
from sklearn.datasets import load_iris

def load_dataset():
    iris = load_iris()
    X = iris['data']
    y = iris['target']


    n_samples, n_features = X.shape

    print('Number of samples:', n_samples)
    print('Number of features:', n_features)

    return X, y

