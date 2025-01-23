from neural_network.gcpu import gcpu

class BinaryCrossEntropyLoss:
    def gradient(self, y_pred, y_true) -> gcpu.ndarray:
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-9)
    
    def loss(self, y_pred, y_true) -> float:
        return -gcpu.mean(
            y_true * gcpu.log(y_pred + 1e-9) + (1 - y_true) * gcpu.log(1 - y_pred + 1e-9)
        )
    
    def accuracy(self, y_pred, y_true) -> float:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return gcpu.mean(y_pred_binary == y_true)
