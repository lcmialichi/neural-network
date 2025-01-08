from abc import ABC, abstractmethod
from neural_network.core import Activation
import numpy as np

class BaseNetwork(ABC):
    def set_training_mode(self) -> None:
        self._mode = 'train'
    
    def set_test_mode(self) -> None:
        self._mode = 'test'
        
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
        return self.output.get('size')

    def get_output_loss(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return self.output.get('activation').loss(x, z)
    
    def get_output_accuracy(self, x: np.ndarray, z: np.ndarray):
        return self.output.get('activation').accuracy(x, z)
    
    def apply_dropout(self, activations: np.ndarray) -> np.ndarray:
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=activations.shape)
        activations = activations * mask
        activations /= (1 - self.dropout_rate)
        return activations