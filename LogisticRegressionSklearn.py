# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# Generating a pre-made binary dataset with two features
X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=20)

# Initializing a Logistic Regression model
lor = LogisticRegression()

# Training the model using the generated dataset
lor.fit(X, y)

m = -(lor.coef_[0][0] / lor.coef_[0][1])          # Slope of the seperating line
b = -(lor.intercept_ / lor.coef_[0][1])           # Intercept of the separating line

# Printing the slope and intercept values
print(m)
print(b)

# Generating x values to plot the decision boundary line
X_input = np.linspace(-3, 3, 100)                # 100 evenly distributed values between -3 and 3
y_input = m * X_input + b                        # Compute corresponding y values using the formula

#plotting
plt.figure(figsize=(10, 8))
plt.plot(X_input, y_input, color='red', linewidth=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
plt.ylim(-3, 2)
plt.show()
