from neural_network.core import Activation
from neural_network.gcpu import gcpu

class Softmax(Activation):
    def activate(self, x, alpha = None):
        exp_z = gcpu.exp(x - gcpu.max(x, axis=1, keepdims=True))
        return exp_z / gcpu.sum(exp_z, axis=1, keepdims=True)

    def derivate(self, x, alpha = None):
        return x
    
