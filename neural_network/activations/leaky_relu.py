from neural_network.core import Activation
import numpy as np

class LeakyRelu(Activation):
    def __init__(self, alpha: float = 0.01, delta: float = 0.1):
        self.alpha = alpha
        self.delta = delta

    def activate(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivate(self, x):
        return np.where(x > 0, 1, self.alpha)

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
      
    def accuracy(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true) < self.delta)

