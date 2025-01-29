from neural_network.core import Activation
from neural_network.loss import LossFunction
from neural_network.core import Initialization
from neural_network.core.optimizer import Optimizer

class Output:
    
    _activation: None | Activation = None
    _loss_function: None | LossFunction = None
    _initializer: None | Initialization = None
    _optimizer: None | Optimizer = None
    _bias: list = []
    _weights: list = []
    
    def __init__(self, size: int):
        self.size = size

    def activation(self, activation: Activation):
        self._activation = activation

    def get_activation(self) -> "Activation":
        return self._activation
    
    def has_activation(self) -> bool:
        return self._activation is not None
    
    def loss_function(self, loss_function: LossFunction):
        self._loss_function = loss_function

    def get_loss_function(self) -> "LossFunction":
        return self._loss_function
    
    def has_loss_function(self) -> bool:
        return self._loss_function is not None
    
    def bias(self):
        return self._bias
    
    def update_bias(self, bias):
        self._bias = bias
    
    def weights(self):
        return self._weights
    
    def update_weights(self, weights):
        self._weights = weights

    def initializer(self, initializer: Initialization):
        self._initializer = initializer
        
    def has_optimizer(self) -> bool:
        return self._optimizer is not None

    def get_optimizer(self) -> bool:
        return self._optimizer
    
    def optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer

    def initialize(self, input_size: int) -> None:
        self._weights = self._initializer.generate_layer(input_size, self.size)
        self._bias = self._initializer.generate_layer_bias(self.size)