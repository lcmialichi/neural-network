from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    
    @abstractmethod
    def __init__(self, learning_rate: float):
        self._learning_rate = learning_rate
        
    @abstractmethod
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        pass
    
    def set_learning_rate(self, lr: float) -> None:
        self._learning_rate = lr
        
    def get_learning_rate(self) -> float:
        return self._learning_rate 