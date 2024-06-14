# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:40:10 2024

@author: appar
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, z):
        return np.heaviside(z, 0)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros((n_features))
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(len(X)):
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(z)
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred)

        # Move the return statement out of the loop
        return self.weights, self.bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, (0, 1)]  # petal length, petal width
y = (iris.target == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

perceptron = Perceptron(0.001, 100)
perceptron.fit(X_train, y_train)
pred = perceptron.predict(X_test)

print("Accuracy:", accuracy_score(pred, y_test))
print("Classification Report:")
print(classification_report(pred, y_test, digits=2))

sk_perceptron = Perceptron(0.001, 100)
sk_perceptron.fit(X_train, y_train)
sk_perceptron_pred = sk_perceptron.predict(X_test)
# Accuracy
print("Accuracy:",accuracy_score(sk_perceptron_pred, y_test))