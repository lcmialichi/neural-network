from abc import ABC, abstractmethod
from neural_network.gcpu import driver
from typing import Callable, Union

class BaseTrainer(ABC):
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor
        self._history: list = []

    @abstractmethod
    def train(self,
            epochs: int = 10, 
            plot: Union[None, Callable] = None,
            callbacks: list = []
            ):
        pass

    def history(self, key: None|str = None):
        if key:
            return self._history[key]
        
        return self._history
    
    def _add_history(self, data: dict):
        self._history.append(data)
