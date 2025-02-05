from neural_network.core import Activation
from neural_network.gcpu import driver

class Softmax(Activation):
    def activate(self, x, alpha = None):
        exp_z = driver.gcpu.exp(x - driver.gcpu.max(x, axis=1, keepdims=True))
        return exp_z / driver.gcpu.sum(exp_z, axis=1, keepdims=True)

    def derivate(self, x, alpha = None):
        return 1
    
