from neural_network.core import Activation
from neural_network.gcpu import driver

class Softmax(Activation):
    def __init__(self, axis=1):
        self.axis = axis

    def activate(self, x, alpha=None):
        x_max = driver.gcpu.max(x, axis=self.axis, keepdims=True)
        exp_z = driver.gcpu.exp(x - x_max)
        sum_exp = driver.gcpu.sum(exp_z, axis=self.axis, keepdims=True)
        return exp_z / sum_exp

    def derivate(self, x, alpha = None):
        return 1
    
