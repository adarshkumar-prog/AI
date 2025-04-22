# Importing the necessary libraries
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# Generating a 2D binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=20)

# Defined perceptron function using the step function
def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)        # Inserting 1 in the matrix
    weights = np.ones(X.shape[1])         # Initialize weights to ones
    lr = 0.1                               # Learning rate

    # epochs = 1000
    for i in range(1000):
        j = np.random.randint(0, 100)     # Randomly selecting one training data
        y_hat = step(np.dot(weights, X[j]))  # Compute prediction using step function
        weights = weights + lr * (y[j] - y_hat) * X[j]  # Updating weights using perceptron rule

    return weights[0], weights[1:]        # Return intercept_ and coef_

def step(z):
    return 1 if z > 0 else 0


intercept_, coef_ = perceptron(X, y)

# Calculating slope (m) and intercept (b) of the seperating line: y = mx + b
m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])

# Print slope and intercept
print(m)
print(b)


X_input = np.linspace(-3, 3, 100)
y_input = m * X_input + b

plt.figure(figsize=(10, 6))
plt.plot(X_input, y_input, color='red', linewidth=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.ylim(-3, 2)
plt.show()
