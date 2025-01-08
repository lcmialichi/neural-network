from neural_network.core import Activation
import numpy as np

class Sigmoid(Activation):
    def activate(self, x, alpha = None):
        return 1 / (1 + np.exp(-x))

    def derivate(self, x, alpha = None):
        return x * (1 - x)
    
    def loss(self, y_pred, y_true):
        epsilon = 1e-9 
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))  
      
    def accuracy(self, y_pred, y_true):
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions == y_true)