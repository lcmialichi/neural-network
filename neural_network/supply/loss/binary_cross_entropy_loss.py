from neural_network.gcpu import driver

class BinaryCrossEntropyLoss:
    def gradient(self, y_pred, y_true):
        return (y_pred - y_true)

    def loss(self, y_pred, y_true) -> float:
        y_pred = driver.gcpu.clip(y_pred, 1e-9, 1 - 1e-9)
        return -driver.gcpu.sum(y_true * driver.gcpu.log(y_pred)) / y_true.shape[0]
        
    def accuracy(self, y_pred, y_true) -> float:
        y_pred_labels = driver.gcpu.argmax(y_pred, axis=1)
        y_true_labels = driver.gcpu.argmax(y_true, axis=1)
        return driver.gcpu.mean(y_pred_labels == y_true_labels)