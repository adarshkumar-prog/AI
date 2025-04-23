# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Generating a synthetic dataset for binary classification
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, hypercube=False, class_sep=20)

# Gradient descent-based logistic regression training function
def gd(X, y):
    X = np.insert(X, 0, 1, axis=1)      
    weights = np.ones(X.shape[1])         # Initializing weights to 1
    lr = 0.1                               # Learning rate

    # Performing the gradient descent for 4000 iterations
    for i in range(4000):
        y_hat = sigmoid(np.dot(X, weights))
        weights = weights + lr * (np.dot((y - y_hat), X) / X.shape[0])  # Updating the weights using gradient

    return weights[0], weights[1:]       # Returning the intercept and the coefficients

# Sigmoid function 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))        

# Training the model using the gradient descent method
intercept_, coef_ = gd(X, y)

# Calculating the slope (m) and intercept (b) y = mx + b
m = -(coef_[0] / coef_[1])               # Slope of the seperating line
b = -(intercept_ / coef_[1])             # Intercept of the separating line

# Generating x values to plot the decision boundary line
X_input = np.linspace(-3, 3, 100)        # 100 evenly distributed values between -3 and 3
y_input = m * X_input + b               # Compute corresponding y values using the formula

# Plotting section
plt.figure(figsize=(10, 6))
plt.plot(X_input, y_input, color='brown', linewidth=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
plt.ylim(-3, 3)
plt.show()
