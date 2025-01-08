from neural_network.core import Activation
import numpy as np

class Relu(Activation):
    
    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def activate(self, x, alpha = None):
        return np.maximum(0, x)

    def derivate(self, x, alpha = None):
        return (x > 0).astype(float)

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)  
      
    def accuracy(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true) < self.delta)