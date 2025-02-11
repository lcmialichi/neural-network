from neural_network.core.base_network import BaseNetwork

class WarmUpScheduler:
    def __init__(self, initial_lr: float, target_lr: float, warmup_epochs: int):
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def __call__(self, model: BaseNetwork):
        if self.current_epoch < self.warmup_epochs:
            new_lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.current_epoch / self.warmup_epochs)
            model.set_learning_rate(new_lr)
            self.current_epoch += 1
