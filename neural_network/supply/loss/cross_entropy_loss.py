from neural_network.gcpu import driver

class CrossEntropyLoss:
    def gradient(self, y_pred, y_true):
        return y_pred - y_true
    
    def loss(self, y_pred, y_true) -> float:
        return -driver.gcpu.mean(driver.gcpu.sum(y_true * driver.gcpu.log(y_pred + 1e-9), axis=1))
    
    def accuracy(self, y_pred, y_true) -> float:
        return driver.gcpu.mean(driver.gcpu.argmax(y_pred, axis=1) == driver.gcpu.argmax(y_true, axis=1))