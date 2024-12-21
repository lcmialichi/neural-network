from neural_network.core import Activation
import numpy as np

class LeakyRelu(Activation):
    def activate(self, x, alpha = 0.1):
        return np.where(x > 0, x, x * alpha)

    def derivate(self, x, alpha = 0.1):
        return np.where(x > 0, 1, alpha)
