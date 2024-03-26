from data import *
from utils import *
from model import KNNClassifer

X_train, X_test , y_train, y_test = generate_dataset()

neighbors = [1,2,3,5,7,11,15,99]
accuracy = []

for i,k in enumerate(neighbors):
    model = KNNClassifer(k)
    model.fit(X_train, y_train)
    predictions =  model.predict(x_test=X_test)
    accuracy.append(check_accuracy(y_pred=predictions,y=y_test,size=y_test.shape[0]))

print(accuracy)