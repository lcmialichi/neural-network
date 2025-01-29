from abc import ABC, abstractmethod
from neural_network.train.base_trainer import BaseTrainer
from neural_network.core.processor import Processor
from neural_network.core.tester import Tester
from neural_network.core.optimizer import Optimizer
from neural_network.gcpu import gcpu
from neural_network.foundation import Output

class BaseNetwork(ABC):
    _output: Output | None = None
    _global_optimizer: Optimizer | None = None

    def set_training_mode(self) -> None:
        self._mode = 'train'
    
    def set_test_mode(self) -> None:
        self._mode = 'test'

    def is_trainning(self) -> bool:
        return self._mode in 'train'
        
    @abstractmethod
    def forward(self, x: gcpu.ndarray):
        pass

    @abstractmethod
    def backward(self, x: gcpu.ndarray, y: gcpu.ndarray, output: gcpu.ndarray):
        pass

    @abstractmethod
    def train(self, x_batch: gcpu.ndarray, y_batch: gcpu.ndarray):
        pass

    @abstractmethod
    def predict(self, x: gcpu.ndarray):
        pass

    def get_processor(self) -> "Processor": 
        return self._processor
    
    def set_processor(self, processor: Processor) -> None: 
        self._processor = processor
        
    def get_learning_rate(self) -> float: 
        return self._global_optimizer.get_learning_rate()
    
    def set_learning_rate(self, val: float) -> None:
        self._global_optimizer.set_learning_rate(val)
    
    @abstractmethod
    def get_trainer(self) -> "BaseTrainer":
        pass

    def get_tester(self) -> Tester:
        return Tester(self, self.get_processor())

    def get_output_size(self) -> int:
        return self.output.get('size')

    def get_output_loss(self, x: gcpu.ndarray, z: gcpu.ndarray) -> gcpu.ndarray:
        return self._output.get_loss_function().loss(x, z)
    
    def get_output_accuracy(self, x: gcpu.ndarray, z: gcpu.ndarray):
        return self._output.get_loss_function().accuracy(x, z)
    
    def _apply_dropout(self, activations: gcpu.ndarray, rate: float) -> gcpu.ndarray:
        assert 0 <= rate < 1, "Dropout rate must be in the range [0, 1)."
        
        retain_prob = 1 - rate
        mask = (gcpu.random.random(size=activations.shape) < retain_prob).astype(activations.dtype)
        
        activations *= mask
        activations /= retain_prob
        
        return activations