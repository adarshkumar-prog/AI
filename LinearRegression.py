from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X,Y = make_regression(n_samples=100, n_features = 1, n_informative = 1, n_targets = 1, noise = 20)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)
lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

class GDRegressor:
    def __init__(self, learning_rate, epochs):
        self.m = 840
        self.b = 1200
        self.lr = learning_rate
        self.epochs = epochs
    def fit(self, X, Y):
        for i in range(self.epochs):
            loss_slope_b = -2*(np.sum(Y-self.m*X.ravel()-self.b))
            loss_slope_m = -2*(np.sum((Y - self.m*X.ravel()-self.b)*X.ravel()))
            self.b = self.b - (self.lr*loss_slope_b)
            self.m = self.m - (self.lr*loss_slope_m)

    def predict(self, X):
        return self.m*X + self.b

gd = GDRegressor(0.001,100)

gd.fit(X_train, y_train)

print(y_pred[0])

print(gd.predict(X_test[0]))

