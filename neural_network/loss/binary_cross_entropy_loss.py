from neural_network.gcpu import gcpu

class BinaryCrossEntropyLoss:
    def gradient(self, y_pred, y_true) -> gcpu.ndarray:
        return y_pred - gcpu.argmax(y_true, axis=-1, keepdims=True)

    def loss(self, y_pred, y_true) -> float:
        return -gcpu.mean(
            y_true * gcpu.log(y_pred + 1e-9) + (1 - gcpu.argmax(y_true, axis=-1, keepdims=True)) * gcpu.log(1 - y_pred + 1e-9)
        )
    
    def accuracy(self, y_pred, y_true) -> float:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return gcpu.mean(y_pred_binary == gcpu.argmax(y_true, axis=-1, keepdims=True))
