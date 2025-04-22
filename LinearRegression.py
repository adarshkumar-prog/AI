# Importing the necessary libraries
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Generating a dataset on which we will implement LinearRegression
X, Y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20)

# Import LinearRegression model and a function for splitting the data into training and testing data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and train a Linear Regression model from sklearn
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict the outputs for test data using the trained sklearn model
y_pred = lr.predict(X_test)

# Creating a linear regression model using Gradient Descent
class GDRegressor:
    def __init__(self, learning_rate, epochs):
        self.m = 840            # Initial slope (m) with any value we want
        self.b = 1200           # Initial intercept (b) with any value we want
        self.lr = learning_rate # Learning rate
        self.epochs = epochs    # Number of iterations for training

    def fit(self, X, Y):
        # Performing the gradient descent for a given number of epochs
        for i in range(self.epochs):
            #derivative of loss function
            loss_slope_b = -2 * np.sum(Y - self.m * X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((Y - self.m * X.ravel() - self.b) * X.ravel())

            # Updating the b and m using gradient descent algorithm
            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)

    def predict(self, X):
        return self.m * X + self.b

# Instantiate owr own gradient descent regression
gd = GDRegressor(learning_rate=0.001, epochs=100)

# Training the data
gd.fit(X_train, y_train)

# Compare predictions: sklearn vs. self-made gradient descent on test data points
print(y_pred[0])                      # Prediction from sklearn model (first test sample)
print(gd.predict(X_test[0]))         # Prediction from own model (first test sample)
print(y_pred[4])                      # Prediction from sklearn model (fifth test sample)
print(gd.predict(X_test[4]))         # Prediction from own model (fifth test sample)

print(type(y_test))

# Initialize an empty array to store predictions from the custom model
y_pred_GD = np.empty(X_test.shape[0])

# Manually predict each test sample using custom model and storing the prediction values
for i in range(X_test.shape[0]):
    y_pred_GD[i] = gd.predict(X_test[i])

# Evaluate the accuracy of the self-made model on the test data
print("Acurracy", r2_score(y_pred_GD, y_test))
