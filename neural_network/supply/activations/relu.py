from neural_network.core import Activation
from neural_network.gcpu import driver

class Relu(Activation):
    
    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def activate(self, x, alpha = None):
        return driver.gcpu.maximum(0, x)

    def derivate(self, x, alpha = None):
        return driver.gcpu.where(x > 0, 1.0, 0.0)