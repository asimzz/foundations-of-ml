from utils import *

class KNNClassifer:
    def __init__(self, K):
        self.K = K
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def __predict_labels(self, distance):
        # Initialize y_pred
        n_test = distance.shape[1]
        y_pred = np.zeros((n_test, 1))
        #predict the label for each example in x_test
        for i in range(n_test):
            #get the closest k examples
            knn_indices = np.argsort(distance[:,i])[:self.K]
            #Get the labels for the closest k
            knn_labels = np.array([self.y_train[j,0] for j in knn_indices])
            #Use the majority vote to predict the label
            y_pred[i] = np.argmax(np.bincount(knn_labels))

        return y_pred
    
    def predict(self, x_test):
        distances = get_distance(self.X_train, x_test)
        predictions = self.__predict_labels(distances)

        return predictions
