from abc import ABC, abstractmethod
from neural_network.gcpu import driver

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    @abstractmethod
    def update(self, param_name: str, param, grad, weight_decay: bool = True):
        pass
    
    def set_learning_rate(self, lr: float) -> None:
        self.learning_rate = lr
        
    def get_learning_rate(self) -> float:
        return self.learning_rate 