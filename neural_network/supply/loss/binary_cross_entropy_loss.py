from neural_network.gcpu import driver

class BinaryCrossEntropyLoss:
    def gradient(self, y_pred, y_true):
        return y_pred - driver.gcpu.argmax(y_true, axis=-1, keepdims=True)

    def loss(self, y_pred, y_true) -> float:
        return -driver.gcpu.mean(
            y_true * driver.gcpu.log(y_pred + 1e-9) + (1 - driver.gcpu.argmax(y_true, axis=-1, keepdims=True)) * driver.gcpu.log(1 - y_pred + 1e-9)
        )
    
    def accuracy(self, y_pred, y_true) -> float:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return driver.gcpu.mean(y_pred_binary == driver.gcpu.argmax(y_true, axis=-1, keepdims=True))
