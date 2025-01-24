from abc import ABC, abstractmethod

class Activation(ABC):

    @abstractmethod
    def activate(self, x, alpha = None):
        pass
    
    @abstractmethod
    def derivate(self, x, alpha = None):
        pass
 