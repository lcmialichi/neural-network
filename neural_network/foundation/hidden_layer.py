from neural_network.core import Activation
from neural_network.core.optimizer import Optimizer
from neural_network.core import Initialization
from neural_network.normalization import BatchNormalization

class HiddenLayer:
    def __init__(
        self,
        size: int, 
        dropout: float | None = None,
        ):
        self._initializer: None | Initialization = None
        self._activation: None | Activation = None
        self._optimizer: None | Optimizer = None
        self._batch_normalization: None | BatchNormalization = None
        self._bias = []
        self._weights = []
        self.size = size
        self._dropout = dropout

    def has_dropout(self) -> bool:
        return self._dropout is not None
    
    def dropout(self, rate: float) -> None:
        self._dropout = rate

    def get_dropout(self):
        return self._dropout
    
    def initializer(self, initializer: Initialization):
        self._initializer = initializer

    def activation(self, activation: Activation):
        self._activation = activation

    def get_activation(self) -> "Activation":
        return self._activation
    
    def has_activation(self) -> bool:
        return self._activation is not None
    
    def batch_normalization(self, gama: float = 1.0, beta: float = 0.0, momentum: float = 0.9) -> None:
        self._batch_normalization = BatchNormalization(self.size, gama=gama, beta=beta, momentum=momentum)

    def get_batch_normalization(self) -> "BatchNormalization":
        return self._batch_normalization
    
    def has_batch_normalization(self)-> bool:
        return self._batch_normalization is not None
    
    def has_optimizer(self) -> bool:
        return self._optimizer is not None

    def get_optimizer(self) -> bool:
        return self._optimizer
    
    def optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer
    
    def bias(self):
        return self._bias
    
    def update_bias(self, bias):
        self._bias = bias
    
    def weights(self):
        return self._weights
    
    def update_weights(self, weights):
        self._weights = weights
        
    def initialize(self, input_size: int) -> None:
        self._weights = self._initializer.generate_layer(input_size, self.size)
        self._bias = self._initializer.generate_layer_bias(self.size)