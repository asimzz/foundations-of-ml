# Logistic Regression Implementation

This repository contains an implementation of logistic regression, split into separate modules for better organization and maintainability.

## Overview

The implementation is split into four modules:

1.  **main.py**: This file serves as the entry point to the program. It contains the main logic for loading data, training the logistic regression model, and evaluating its performance.

2.  **model.py**: This module contains the `LogisticRegression` class, which implements logistic regression for binary classification tasks. The class provides methods for training the model (`fit`), making predictions (`predict`), and predicting probabilities (`predict_proba`).

The logistic regression model is initialized with weights (`self.w`) set to `None`.

- `add_ones`: Adds a column of ones to the feature matrix `x` to incorporate bias.

- `sigmoid`: Calculates the sigmoid activation function of the linear combination of features and weights.

<p  align="center">  <img  src="https://latex.codecogs.com/svg.latex?%5Csigma%28z%29%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-z%7D%7D">  </p>

- `predict_proba`: Predicts the probabilities of the positive class.

- `predict`: Predicts the binary class labels (0 or 1) by applying a threshold of 0.5 to the predicted probabilities.

- `fit`: Trains the logistic regression model using gradient descent. It iterates over the dataset for a specified number of epochs (`n_epochs`), updating the weights (`self.w`) based on the cross-entropy loss and the gradient of the loss with respect to the weights.

The `cross_entropy` function used for calculating the loss is imported from the `utils.py` module.

3.  **data.py**: This module provides functions for generating synthetic data for training and testing the logistic regression model. It utilizes the `make_classification` function from scikit-learn to create a synthetic dataset with specified features and classes. The `generate_data` function splits the generated dataset into training and testing sets using `train_test_split` from scikit-learn, returning the feature matrices and corresponding labels.

4.  **utils.py**: This module contains utility functions used in the logistic regression implementation.

- The `cross_entropy` function calculates the cross-entropy loss between the true labels (`y_true`) and predicted probabilities (`y_pred`). The cross-entropy formula used is:

<p  align="center">

<img  src="https://latex.codecogs.com/svg.latex?%5Ctext%7BCross-Entropy%7D%20%3D%20-%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Cleft%28%20y_%7B%5Ctext%7Btrue%7D%2C%20i%7D%20%5Clog%28y_%7B%5Ctext%7Bpred%7D%2C%20i%7D%29%20+%20%281%20-%20y_%7B%5Ctext%7Btrue%7D%2C%20i%7D%29%20%5Clog%281%20-%20y_%7B%5Ctext%7Bpred%7D%2C%20i%7D%29%20%5Cright%29">

</p>

- The `check_accuracy` function computes the accuracy of the model's predictions by comparing the true labels with the predicted labels (`y_pred`), returning the accuracy in percentage.

## Usage

To use this implementation, follow these steps:

1. Clone the repository to your local machine:

```bash
git  clone  https://github.com/asimzz/foundations-of-ml.git

cd  logistic-regression
```

2. Run the main script:

```bash
python  main.py
```
