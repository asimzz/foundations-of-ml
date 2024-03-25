from tqdm import trange
from data import train_test_split
from evaluations import *
from model import SingleLayerNN


X_train, Y_train, X_test, Y_test = train_test_split()

"""## Initialize parameters"""
model = SingleLayerNN(h0=2, h1=10, h2=1)


alpha = 0.1
n_epochs = 10000
train_loss = []
test_loss = []
for i in trange(n_epochs):
  ## forward pass
  A2, Z2, A1, Z1 = model.forward_pass(X_train)
  ## backward pass
  dW1, dW2, db1, db2 = model.backward_pass(X_train, Y_train, A2, A1, Z1)
  ## update parameters
  model.update(dW1, dW2, db1, db2, alpha)

  ## save the train loss
  train_loss.append(loss(A2, Y_train))

  ## compute test loss
  A2_test ,_ ,_ ,_ = model.forward_pass(X_test)
  test_loss.append(loss(A2_test, Y_test))

  # plot boundary
  if i %1000 == 0:
   print(f"Epoch {i}/10000:")
   print(f"Train Loss = {train_loss[i]}")
   print(f"Test Loss = {test_loss[i]}")

y_pred = model.predict(X_train)
train_accuracy = accuracy(y_pred, Y_train)
print (f"Train Accuracy : {int(train_accuracy)}%")

y_pred = model.predict(X_test)
test_accuracy = accuracy(y_pred, Y_test)
print (f"Test Accuracy : {int(test_accuracy)}%")