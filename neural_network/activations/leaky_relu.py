from neural_network.core import Activation
from neural_network.gcpu import gcpu

class LeakyRelu(Activation):
    def __init__(self, alpha: float = 0.01, delta: float = 0.1):
        self.alpha = alpha
        self.delta = delta

    def activate(self, x):
        return gcpu.where(x > 0, x, self.alpha * x)

    def derivate(self, x):
        return gcpu.where(x > 0, 1, self.alpha)

    def loss(self, y_pred, y_true):
        return gcpu.mean((y_pred - y_true) ** 2)
      
    def accuracy(self, y_pred, y_true):
        return gcpu.mean(gcpu.abs(y_pred - y_true) < self.delta)

