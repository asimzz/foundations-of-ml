from data import generate_data
from model import LinearRegression


Xtrain, ytrain = generate_data()
model = LinearRegression()

model.fit(Xtrain, ytrain)



