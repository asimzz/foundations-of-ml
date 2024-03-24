import numpy as np
# generate data

def generate_data():
    var = 0.2
    N = 800
    class_0_a = var * np.random.randn(N//4,2)
    class_0_b =var * np.random.randn(N//4,2) + (2,2)

    class_1_a = var* np.random.randn(N//4,2) + (0,2)
    class_1_b = var * np.random.randn(N//4,2) +  (2,0)

    X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
    Y = np.concatenate([np.zeros((N//2,1)), np.ones((N//2,1))])


    # shuffle the data
    rand_perm = np.random.permutation(N)

    X = X[rand_perm, :]
    Y = Y[rand_perm, :]

    X = X.T
    Y = Y.T

    return X, Y, N


# train test split
def train_test_split():
    ratio = 0.8
    X,Y,N = generate_data()
    X_train = X[:, :int (N*ratio)]
    Y_train = Y[:, :int (N*ratio)]

    X_test = X[:, int (N*ratio):]
    Y_test = Y[:, int (N*ratio):]

    return X_train, Y_train, X_test, Y_test