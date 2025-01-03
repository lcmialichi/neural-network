from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union

class BaseTrainer(ABC):
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def train(self, data_source, epochs: int, batch_size: int, plot: Union[None, Callable] = None):
        pass

    @staticmethod
    def compute_loss(y_pred, y_true):
        exp_logits = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return -np.mean(np.sum(y_true * np.log(softmax + 1e-9), axis=1))

    @staticmethod
    def compute_accuracy(y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
