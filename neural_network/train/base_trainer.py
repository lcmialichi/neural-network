from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union

class BaseTrainer(ABC):
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def train(self, data_source, epochs: int, batch_size: int, plot: Union[None, Callable] = None):
        pass
