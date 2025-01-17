from neural_network.core import Activation
from neural_network.gcpu import gcpu

class Sigmoid(Activation):
    def activate(self, x, alpha = None):
        return 1 / (1 + gcpu.exp(-x))

    def derivate(self, x, alpha = None):
        return x * (1 - x)
    
    def loss(self, y_pred, y_true):
        epsilon = 1e-9 
        return -gcpu.mean(y_true * gcpu.log(y_pred + epsilon) + (1 - y_true) * gcpu.log(1 - y_pred + epsilon))  
      
    def accuracy(self, y_pred, y_true):
        predictions = (y_pred > 0.5).astype(int)
        return gcpu.mean(predictions == y_true)