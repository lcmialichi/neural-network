from abc import ABC, abstractmethod
from typing import Tuple, Generator
from neural_network.gcpu import driver

class Pooling(ABC):
    
    stride: int
    shape: tuple[int, int]

    @abstractmethod
    def apply_pooling(self, input):
        pass
    
    @abstractmethod
    def unpooling(self, grad):
        pass
