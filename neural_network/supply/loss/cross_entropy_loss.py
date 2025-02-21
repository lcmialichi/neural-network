from neural_network.gcpu import driver

class CrossEntropyLoss:
    def __init__(self, epsilon=1e-7, reduction='mean'):
        self._epsilon = epsilon
        self.reduction = reduction

    def gradient(self, y_pred, y_true):
        return y_pred - y_true
    
    def loss(self, y_pred, y_true) -> float:
        clipped_y_pred = driver.gcpu.clip(y_pred, self._epsilon, 1 - self._epsilon)
        
        per_sample_loss = -driver.gcpu.sum(y_true * driver.gcpu.log(clipped_y_pred), axis=-1)
        
        if self.reduction == 'mean':
            return driver.gcpu.mean(per_sample_loss)
        elif self.reduction == 'sum':
            return driver.gcpu.sum(per_sample_loss)
        elif self.reduction == 'none':
            return per_sample_loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
    
    def accuracy(self, y_pred, y_true) -> float:
        return driver.gcpu.mean(driver.gcpu.argmax(y_pred, axis=1) == driver.gcpu.argmax(y_true, axis=1))