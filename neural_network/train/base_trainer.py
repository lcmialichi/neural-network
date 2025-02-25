from abc import ABC, abstractmethod
from neural_network.gcpu import driver
from typing import Callable, Union
from neural_network.core.processor import Processor

class BaseTrainer(ABC):
    def __init__(self, model):
        self._model = model
        self._history: list = []

    @abstractmethod
    def train(self,
            processor: Processor,
            epochs: int = 10,
            plot: Union[None, Callable] = None,
            callbacks: list = []
            ):
        pass

    def history(self):
        return self._history
    
    def _add_history(self, data: dict):
        self._history.append(data)
