from neural_network.core.base_network import BaseNetwork

class ReduceLROnPlateau:
    def __init__(self,  factor: float=0.5, patience: int =5, min_lr: float =1e-6):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0

    def __call__(self, model: BaseNetwork, val_loss, val_accuracy):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                new_lr = max(self.min_lr, model.get_learning_rate() * self.factor)
                if new_lr < model.get_learning_rate():
                    model.set_learning_rate(new_lr)
                self.wait = 0
