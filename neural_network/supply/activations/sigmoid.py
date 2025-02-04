from neural_network.core import Activation
from neural_network.gcpu import gcpu

class Sigmoid(Activation):
    def activate(self, x, alpha = None):
        return 1 / (1 + gcpu.exp(-x))

    def derivate(self, x, alpha = None):
        return x * (1 - x)
    