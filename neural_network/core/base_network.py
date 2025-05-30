from abc import ABC, abstractmethod
from neural_network.train.base_trainer import BaseTrainer
from neural_network.core.processor import Processor
from neural_network.core.tester import Tester
from neural_network.core.optimizer import Optimizer
from neural_network.gcpu import driver

class BaseNetwork(ABC):
    _global_optimizer: Optimizer | None = None
    _dropout_mask = None

    def set_training_mode(self) -> None:
        self._mode = 'train'
    
    def set_test_mode(self) -> None:
        self._mode = 'test'

    def is_training(self) -> bool:
        return self._mode in 'train'
        
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x, y, output):
        pass

    @abstractmethod
    def train(self, x_batch, y_batch):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def get_processor(self) -> "Processor": 
        return self._processor
    
    def get_learning_rate(self) -> float: 
        return self._global_optimizer.get_learning_rate()
    
    def set_learning_rate(self, val: float) -> None:
        self._global_optimizer.set_learning_rate(val)
        
    def step(self):
        self._global_optimizer.step()
    
    @abstractmethod
    def get_trainer(self) -> "BaseTrainer":
        pass

    def get_tester(self) -> Tester:
        return Tester(self)

    def get_output_loss(self, x, z):
        return self._loss_function.loss(x, z)
    
    def get_output_accuracy(self, x, z):
        return self._loss_function.accuracy(x, z)
