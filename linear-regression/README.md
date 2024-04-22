# Linear Regression Implementation

This repository contains an implementation of linear regression, split into separate modules for better organization and maintainability.

## Overview

The implementation is split into four modules:

1.  **main.py**: This file serves as the entry point to the program. It contains the main logic for loading data, training the linear regression model, and evaluating its performance.

2.  **model.py**: This module contains the `LinearRegression` class, which implements linear regression for regression tasks. The class provides methods for training the model (`fit`) and performing forward and backward passes.

- `LinearRegression`: Initializes the linear regression model with weights (`self.theta`) set to `None`.

The class includes the following methods:

- `forward_pass(X)`: Performs a forward pass through the linear regression model, computing the predicted values based on the input features `X` and the model parameters `self.theta`. The forward pass is represented as

$$
H(X) = X \cdot  \theta
$$

- `backward_pass(X, y)`: Performs a backward pass through the linear regression model, computing the gradient of the mean squared error loss function with respect to the model parameters `self.theta`.

- `update_param(grads, step_size=0.1)`: Updates the model parameters `self.theta` based on the computed gradients and a specified step size.

- `fit(Xtrain, ytrain, n_epochs=10)`: Trains the linear regression model using gradient descent. It iterates over the dataset for a specified number of epochs (`n_epochs`), updating the model parameters `self.theta` based on the mean squared error loss and the gradient of the loss with respect to the parameters.

The `mean_squared_error` function used for calculating the loss is imported from the `utils.py` module.

3.  **data.py**: This module provides functions for generating synthetic data for training and testing the linear regression model.

- `generate_data()`: Generates synthetic training data for linear regression. It creates a set of input features (`xtrain`) ranging from 0 to 1 with 10 evenly spaced points and corresponding target labels (`ytrain`). Gaussian noise with mean 0 and standard deviation 0.1 is added to the true relationship between `xtrain` and `ytrain`.

4.  **utils.py**: This module contains utility functions used in the linear regression implementation.

- `mean_squared_error(ytrue, ypred)`: Calculates the mean squared error (MSE) between the true labels (`ytrue`) and the predicted labels (`ypred`). The MSE is computed as the average of the squared differences between corresponding elements of `ytrue` and `ypred`.

<p  align="center">
<img  src="https://latex.codecogs.com/svg.latex?%5Ctext%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28y_%7B%5Ctext%7Btrue%7D%2C%20i%7D%20-%20y_%7B%5Ctext%7Bpred%7D%2C%20i%7D%29%5E2">

</p>

## Usage

To use this implementation, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/asimzz/foundations-of-ml.git

cd linear-regression

```

2. Run the main script:

```bash
python main.py
```
