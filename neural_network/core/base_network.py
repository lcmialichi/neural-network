from abc import ABC, abstractmethod
from neural_network.train.base_trainer import BaseTrainer
from neural_network.core.processor import Processor
from neural_network.core.tester import Tester
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

    def get_processor(self) -> "Processor": 
        return self._processor
    
    def set_processor(self, processor: Processor) -> None: 
        self._processor = processor
        
    def get_learning_rate(self) -> float: 
        self.global_optimizer.get_learning_rate()
    
    def set_learning_rate(self, val: float) -> None:
        self.global_optimizer.set_learning_rate(val)
    
    @abstractmethod
    def get_trainer(self) -> "BaseTrainer":
        pass

    def get_tester(self) -> Tester:
        return Tester(self, self.get_processor())

    def get_output_size(self) -> int:
        return self.output.get('size')

    def get_output_loss(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return self.output.get('activation').loss(x, z)
    
    def get_output_accuracy(self, x: np.ndarray, z: np.ndarray):
        return self.output.get('activation').accuracy(x, z)
    
    def _apply_dropout(self, activations: np.ndarray, rate: float) -> np.ndarray:
        retain_prob = 1 - rate
        mask = self.rng.random(size=activations.shape) < retain_prob
        activations = activations * mask
        activations /= retain_prob
        return activations