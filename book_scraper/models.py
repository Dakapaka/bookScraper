import numpy as np
from scipy.special import expit
import pandas as pd

class LinearRegression:
    def __init__(self, lr=0.0001, n_iters=1000, lambda_=0.001):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            error = y_predicted - y
            dw = (1 / n_samples) * (np.dot(X.T, error) + self.lambda_ * self.weights**2)
            db = (1 / n_samples) * np.sum(error)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"Iteration {i}: Weights mean = {np.mean(self.weights)}, Bias = {self.bias}")

            if np.any(np.isnan(self.weights)) or np.isnan(self.bias):
                print("Warning: NaN values detected in weights or bias")
                return 

    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        if np.any(np.isnan(predictions)):
            print("Warning: NaNs detected in predictions")
        return predictions
    
    def score(self, X, y):
        y_predicted = self.predict(X)
        return 1 - (np.sum((y - y_predicted) ** 2) / np.sum((y - y.mean()) ** 2))

class LogisticRegression:
    def __init__(self, lr=0.01, num_iterations=1000, multi_class='ovr'):
        self.lr = lr
        self.num_iterations = num_iterations
        self.multi_class = multi_class
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))
        
        if self.multi_class == 'ovr':
            self.weights = np.zeros((num_classes, n))
            self.biases = np.zeros(num_classes)
            
            for i in range(num_classes):
                y_binary = (y == i).astype(int)
                w = np.zeros(n)
                b = 0

                for _ in range(self.num_iterations):
                    Z = np.dot(X, w) + b
                    A = expit(Z)
                    
                    dw = (1 / m) * np.dot(X.T, (A - y_binary))
                    db = (1 / m) * np.sum(A - y_binary)
                    
                    w -= self.lr * dw
                    b -= self.lr * db
                
                self.weights[i] = w
                self.biases[i] = b
        
        elif self.multi_class == 'multinomial':
            self.weights = np.zeros((num_classes, n))
            self.biases = np.zeros(num_classes)
            
            for _ in range(self.num_iterations):
                Z = np.dot(X, self.weights.T) + self.biases
                A = expit(Z)
                
                dZ = A - pd.get_dummies(y).values
                dw = (1 / m) * np.dot(dZ.T, X)
                db = (1 / m) * np.sum(dZ, axis=0)
                
                self.weights -= self.lr * dw
                self.biases -= self.lr * db

    def predict(self, X):
        if self.multi_class == 'ovr':
            num_classes = self.weights.shape[0]
            predictions = np.zeros((X.shape[0], num_classes))
            
            for i in range(num_classes):
                Z = np.dot(X, self.weights[i]) + self.biases[i]
                A = expit(Z)
                predictions[:, i] = A
            
            return np.argmax(predictions, axis=1)
        
        elif self.multi_class == 'multinomial':
            Z = np.dot(X, self.weights.T) + self.biases
            A = expit(Z)
            return np.argmax(A, axis=1)