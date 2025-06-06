from neural_network.core import Activation
from neural_network.gcpu import driver

class LeakyRelu(Activation):
    def __init__(self, alpha: float = 0.01, delta: float = 0.1):
        self.alpha = alpha
        self.delta = delta

    def activate(self, x):
        return driver.gcpu.where(x > 0, x, self.alpha * x)

    def derivate(self, x):
        return driver.gcpu.where(x > 0, 1, self.alpha)


