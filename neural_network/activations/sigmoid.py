from neural_network.core import Activation
import numpy as np

class Sigmoid(Activation):
    def activate(self, x, alpha = None):
        return 1 / (1 + np.exp(-x))

    def derivate(self, x, alpha = None):
        return x * (1 - x)