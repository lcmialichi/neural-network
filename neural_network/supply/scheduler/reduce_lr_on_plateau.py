from neural_network.core.base_network import BaseNetwork

class ReduceLROnPlateau:
    def __init__(self, 
                 monitor: str = 'val_loss', 
                 factor: float = 0.5, 
                 patience: int = 5, 
                 min_lr: float = 1e-6, 
                 threshold: float = 0.0,
                 mode: str = 'min',
                 cooldown: int = 0,
                 verbose: bool = False):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.mode = mode
        self.cooldown = cooldown
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.cooldown_counter = 0

    def __call__(self, model: BaseNetwork, metrics: dict):
        data = metrics.get(self.monitor)
        if data is None:
            self.print_if_verbose(f"unknown metric {self.monitor}.")
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        improvement = (data < self.best_value - self.threshold) if self.mode == 'min' else (data > self.best_value + self.threshold)

        if improvement:
            self.best_value = data
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = model.get_learning_rate()
                new_lr = max(self.min_lr, current_lr * self.factor)
                if new_lr < current_lr:
                    model.set_learning_rate(new_lr)
                    self.print_if_verbose(f"Reducing learning rate from {current_lr} to {new_lr} (metric {self.monitor}: {data})")
                    self.cooldown_counter = self.cooldown
                self.wait = 0

        
    def print_if_verbose(self, text):
        if self.verbose:
            print(text, flush=True)
