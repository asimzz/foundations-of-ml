
from data import generate_data
from utils import check_accuracy
from model import LogisticRegression

X_train, X_test, y_train, y_test = generate_data()
model = LogisticRegression()
model.fit(X_train, y_train, 0.01)

print(" ")

ypred_train = model.predict(X_train)
acc = check_accuracy(y_train,ypred_train)
print(f"The training accuracy is: {acc}")
print(" ")

ypred_test = model.predict(X_test)
acc = check_accuracy(y_test,ypred_test)
print(f"The test accuracy is: {acc}")