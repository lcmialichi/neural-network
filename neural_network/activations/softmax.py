from neural_network.core import Activation
import numpy as np

class Softmax(Activation):
    def activate(self, x, alpha = None):
        exp_z = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def derivate(self, x, alpha = None):
        raise RuntimeError('softmax derivative not implemented yet')
    
    def loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
    
    def accuracy(self, y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
