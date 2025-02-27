from neural_network.gcpu import driver

class BinaryCrossEntropyLoss:
    def _convert_labels_and_predictions(self, y_pred, y_true):
        if y_pred.shape[-1] > 1:
            y_pred = y_pred[:, 1:2]
        if y_true.shape[-1] > 1:
            y_true = y_true[:, 1:2]
        return y_pred, y_true

    def gradient(self, y_pred, y_true):
        y_pred, y_true = self._convert_labels_and_predictions(y_pred, y_true)
        return y_pred - y_true

    def loss(self, y_pred, y_true) -> float:
        y_pred, y_true = self._convert_labels_and_predictions(y_pred, y_true)
        y_pred = driver.gcpu.clip(y_pred, 1e-9, 1 - 1e-9)
        return -driver.gcpu.mean(
            y_true * driver.gcpu.log(y_pred) + (1 - y_true) * driver.gcpu.log(1 - y_pred)
        )

    def accuracy(self, y_pred, y_true) -> float:
        y_pred, y_true = self._convert_labels_and_predictions(y_pred, y_true)
        y_pred_labels = (y_pred > 0.5).astype(int)
        return driver.gcpu.mean(y_pred_labels == y_true)
