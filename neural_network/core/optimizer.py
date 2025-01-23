from abc import ABC, abstractmethod
from neural_network.gcpu import gcpu

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    @abstractmethod
    def update(self, param_name: str, param: gcpu.ndarray, grad: gcpu.ndarray) -> gcpu.ndarray:
        pass
    
    def set_learning_rate(self, lr: float) -> None:
        self.learning_rate = lr
        
    def get_learning_rate(self) -> float:
        return self.learning_rate 