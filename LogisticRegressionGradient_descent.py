from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples = 100, n_features = 2, n_informative = 1, n_redundant = 0, n_classes = 2, n_clusters_per_class = 1, hypercube = False, class_sep = 20)

def gd(X, y):
    X = np.insert(X, 0, 1, axis = 1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    for i in range(4000):
        y_hat = sigmoid(np.dot(X, weights))
        weights = weights + lr*(np.dot((y - y_hat), X)/X.shape[1])

    return weights[0], weights[1:]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

intercept_, coef_ = gd(X, y)

m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])

X_input = np.linspace(-3, 3, 100)
y_input = m*X_input + b

plt.figure(figsize = (10, 6))
plt.plot(X_input, y_input, color = 'brown', linewidth = 3)
plt.scatter(X[:,0], X[:,1], c = y, cmap = 'coolwarm', s = 100)
plt.ylim(-3, 3)
plt.show()