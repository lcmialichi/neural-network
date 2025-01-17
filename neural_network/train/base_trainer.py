from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union

class BaseTrainer(ABC):
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    @abstractmethod
    def train(self,
            epochs: int = 10, 
            plot: Union[None, Callable] = None,
            scheduler = None
            ):
        pass
