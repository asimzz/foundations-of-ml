import numpy as np

def cross_entropy(y_true, y_pred):
    loss = - np.mean(y_true * np.log(y_pred) + (1-y_true)* np.log(1-y_pred))
    return loss

def check_accuracy(y_true, y_pred):
    acc = np.mean(y_true.reshape(-1,1)==y_pred) * 100
    return acc