from neural_network.initializations import He
from neural_network.normalization import BatchNormalization
from neural_network.pooling import MaxPooling, AvgPooling
from neural_network.core import Activation, Initialization
from neural_network.core.dropout import Dropout
from neural_network.core.pooling import Pooling
from neural_network.core.optimizer import Optimizer
from typing import Callable

class Kernel:
    
    _filters: list = []
    _bias: list = []
    _dropout: Dropout | None = None
    _initializer: Initialization = He()
    _activation: Activation | None = None
    _batch_normalization: BatchNormalization | None  = None
    _pooling: Pooling | None = None
    _optimizer: Optimizer | None = None
    _clip_gradients: tuple[float, float] | None = None

    def __init__(self,  number: int, shape: tuple[int, int] = (3, 3), stride: int = 1):
        self.number = number
        self.shape = shape
        self.stride = stride

    def has_dropout(self) -> bool:
        return self._dropout is not None
    
    def dropout(self, rate: float) -> None:
        self._dropout = Dropout(rate)

    def get_dropout(self) -> "Dropout":
        return self._dropout

    def has_optimizer(self) -> bool:
        return self._optimizer is not None

    def get_optimizer(self) -> bool:
        return self._optimizer
    
    def optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer

    def initializer(self, initializer: Initialization):
        self._initializer = initializer

    def has_pooling(self) -> bool:
        return self._pooling is not None
    
    def get_pooling(self) -> "Pooling":
        return self._pooling

    def max_pooling(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self._pooling = MaxPooling(shape=shape, stride=stride)
    
    def avg_pooling(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self._pooling = AvgPooling(shape=shape, stride=stride)

    def activation(self, activation: Activation):
        self._activation = activation

    def get_activation(self) -> "Activation":
        return self._activation
    
    def has_activation(self) -> bool:
        return self._activation is not None

    def batch_normalization(self, gama: float = 1.0, beta: float = 0.0, momentum: float = 0.9) -> None:
        self._batch_normalization = BatchNormalization(self.number, gama=gama, beta=beta, momentum=momentum)

    def get_batch_normalization(self) -> "BatchNormalization":
        return self._batch_normalization
    
    def has_batch_normalization(self)-> bool:
        return self._batch_normalization is not None

    def filters(self):
        return self._filters
    
    def update_filters(self, filters):
        self._filters = filters

    def bias(self):
        return self._bias

    def update_bias(self, bias):
        self._bias = bias
        
    def initialize(self, channels: int) -> None:
        self._filters = self._initializer.kernel_filters(self.number, self.shape, channels)
        self._bias = self._initializer.kernel_bias(self.number)
        
    def tap(self, callback: Callable[["Kernel"], None]):
        if callable(callback):
            callback(self)
        return self
    
    def clip_gradients(self, min: float = -1e1, max: float = 1e1) -> None:
        self._clip_gradients = (min, max)

    def has_gradients_clipped(self) -> bool:
        return self._clip_gradients is not None

    def get_clip_gradients(self) -> None|tuple[float, float]:
        return self._clip_gradients