from neural_network.gcpu import gcpu

class CrossEntropyLoss:
    def gradient(self, y_pred, y_true) -> gcpu.ndarray:
        return y_pred - y_true
    
    def loss(self, y_pred, y_true) -> float:
        return -gcpu.mean(gcpu.sum(y_true * gcpu.log(y_pred + 1e-9), axis=1))
    
    def accuracy(self, y_pred, y_true) -> float:
        return gcpu.mean(gcpu.argmax(y_pred, axis=1) == gcpu.argmax(y_true, axis=1))