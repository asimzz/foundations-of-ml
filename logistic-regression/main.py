from sklearn.model_selection import train_test_split
from data import generate_data
from model import LogisticRegression

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8)
model = LogisticRegression()
model.fit(X_train, y_train, 0.01)

print(" ")

ypred_train = model.predict(X_train)
acc = model.check_accuracy(y_train,ypred_train)
print(f"The training accuracy is: {acc}")
print(" ")

ypred_test = model.predict(X_test)
acc = model.check_accuracy(y_test,ypred_test)
print(f"The test accuracy is: {acc}")