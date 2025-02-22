from neural_network.core import Activation
from neural_network.gcpu import driver

class Sigmoid(Activation):
    def activate(self, x, alpha = None):
        return 1 / (1 + driver.gcpu.exp(-x))

    def derivate(self, x, alpha = None):
        return 1
    