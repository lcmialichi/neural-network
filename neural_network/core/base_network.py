from abc import ABC, abstractmethod
from neural_network.core import Activation
import numpy as np

class BaseNetwork(ABC):
    def __init__(self, config: dict):
        self.learning_rate = config.get('learning_rate', 0.01)
        self.regularization_lambda = config.get('regularization_lambda', 0.01)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.hidden_size: int = config.get('hidden_size', 128)
        self.layers_number = config.get('layers_number', 2)
        self.output_size = config.get('output_size', 10)

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    @abstractmethod
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        pass

    @abstractmethod
    def train(self, x_batch: np.ndarray, y_batch: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray):
        pass
    
    @abstractmethod
    def get_trainer(self):
        pass
    
    def get_output_size(self) -> int:
        return self.output_size

    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def apply_dropout(self, activations: np.ndarray) -> np.ndarray:
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=activations.shape)
        activations = activations * mask
        activations /= (1 - self.dropout_rate)
        return activations