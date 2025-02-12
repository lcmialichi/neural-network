from neural_network.gcpu import driver

class BinaryCrossEntropyLoss:
    def gradient(self, y_pred, y_true):
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-9)

    def loss(self, y_pred, y_true) -> float:
        return -driver.gcpu.mean(
            y_true * driver.gcpu.log(y_pred + 1e-9) + (1 - y_true) * driver.gcpu.log(1 - y_pred + 1e-9)
        )
    
    def accuracy(self, y_pred, y_true) -> float:
        y_pred_binary = driver.gcpu.argmax(y_pred, axis=-1)
        y_true_binary = driver.gcpu.argmax(y_true, axis=-1)
        return driver.gcpu.mean(y_pred_binary == y_true_binary)
