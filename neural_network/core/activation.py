from abc import ABC, abstractmethod

class Activation(ABC):

    @abstractmethod
    def activate(self, x, alpha = None):
        pass
    
    @abstractmethod
    def derivate(self, x, alpha = None):
        pass
    
    @abstractmethod
    def loss(self, x, alpha = None):
        pass
    
    @abstractmethod
    def accuracy(self, x, alpha = None):
        pass