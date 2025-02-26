from neural_network.core.base_network import BaseNetwork

class ReduceLROnPlateau:
    def __init__(self, 
        monitor: str = 'val_loss', 
        factor: float=0.5, 
        patience: int =5, 
        min_lr: float =1e-6, 
        threshold: float = 0.1
    ):
        self.factor = factor
        self.monitor = monitor
        self.patience = patience
        self.min_lr = min_lr
        self.best_value = float('inf')
        self.wait = 0
        self.threshold = threshold

    def __call__(self, model: BaseNetwork, metrics: dict):
        data = metrics[self.monitor]
        if data < self.best_value - self.threshold:
            self.best_value = data
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                new_lr = max(self.min_lr, model.get_learning_rate() * self.factor)
                if new_lr < model.get_learning_rate():
                    model.set_learning_rate(new_lr)
                self.wait = 0
