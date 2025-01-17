from neural_network.core import Activation
from neural_network.gcpu import gcpu

class Softmax(Activation):
    def activate(self, x, alpha = None):
        exp_z = gcpu.exp(x - gcpu.max(x, axis=1, keepdims=True))
        return exp_z / gcpu.sum(exp_z, axis=1, keepdims=True)

    def derivate(self, x, alpha = None):
        raise RuntimeError('softmax derivative not implemented yet')
    
    def loss(self, y_pred, y_true):
        return -gcpu.mean(gcpu.sum(y_true * gcpu.log(y_pred + 1e-9), axis=1))
    
    def accuracy(self, y_pred, y_true):
        return gcpu.mean(gcpu.argmax(y_pred, axis=1) == gcpu.argmax(y_true, axis=1))
