import numpy as np
from sklearn.linear_model import LinearRegression


class grad_optimizer:
    def __init__(self, max_steps, coeff, grad_func, ):
        self.max_steps = max_steps
        self.coeff = coeff
        self.grad_func = grad_func

    def run(self, x0):
        x = x0
        for i in range(1, self.max_steps + 1):
            grad = self.grad_func(x)
            grad = grad / np.linalg.norm(grad)
            x_new = x + self.coeff * grad
            if np.linalg.norm(x_new - x) < 1e-6:
                break
            x = x_new
        return x


class MyModel(LinearRegression):
    def __init__(self, weights):
        self.weights = weights

    def predict(self, X):
        return np.dot(X, self.weights.T)
