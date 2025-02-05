from abc import ABC, abstractmethod
from neural_network.gcpu import driver

class LossFunction(ABC):
    @abstractmethod
    def gradient(self, y_pred, y_true):
        pass

    @abstractmethod
    def loss(self, y_pred, y_true) -> float:
        pass

    @abstractmethod
    def accuracy(self, y_pred, y_true) -> float:
        pass