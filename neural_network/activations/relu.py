from neural_network.core import Activation
import numpy as np

class Relu(Activation):
    def activate(self, x, alpha = None):
        return np.maximum(0, x)

    def derivate(self, x, alpha = None):
        return (x > 0).astype(float)
