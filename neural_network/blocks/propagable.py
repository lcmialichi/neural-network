from abc import ABC, abstractmethod

class Propagable (ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, input_layer, y, delta):
        pass

    @abstractmethod
    def boot(self, shape: tuple):
        pass