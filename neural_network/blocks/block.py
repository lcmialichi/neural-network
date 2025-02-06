from neural_network.core.optimizer import Optimizer
from neural_network.core.dropout import Dropout
from neural_network.core.initialization import Initialization
from neural_network.core.activation import Activation
from neural_network.supply import normalization
from .propagable import Propagable
from abc import abstractmethod

class Block(Propagable):
    mode: str | None = None
    padding_type = None
    regularization_lambda: float| None = None
    global_optimizer: Optimizer | None = None
    _optimizer: Optimizer | None = None
    loss_function = None
    _initializer: Initialization | None = None
    _clip_gradients: tuple[float, float] | None = None
    _logits = None
    _dropout: Dropout | None = None

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, input_layer, y, delta):
        pass

    @abstractmethod
    def boot(self, shape: tuple):
        pass
        
    def has_dropout(self) -> bool:
        return self._dropout is not None
    
    def dropout(self, rate: float) -> None:
        self._dropout = Dropout(rate)

    def get_dropout(self) -> "Dropout":
        return self._dropout

    def has_optimizer(self) -> bool:
        return self._optimizer is not None and self.global_optimizer is not None

    def get_optimizer(self) -> "Optimizer":
        return self._optimizer if self._optimizer is not None else self.global_optimizer
    
    def optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer
        
    def initializer(self, initializer: Initialization):
        self._initializer = initializer

    def activation(self, activation: Activation):
        self._activation = activation

    def get_activation(self) -> "Activation":
        return self._activation
    
    def has_activation(self) -> bool:
        return self._activation is not None
    
    def batch_normalization(self, gama: float = 1.0, beta: float = 0.0, momentum: float = 0.9) -> None:
        self._batch_normalization = normalization.BatchNormalization(self.number, gama=gama, beta=beta, momentum=momentum)

    def get_batch_normalization(self) -> "normalization.BatchNormalization":
        return self._batch_normalization
    
    def has_batch_normalization(self)-> bool:
        return self._batch_normalization is not None
    
    def logits(self):
        return self._logits

    def store_logits(self, logits):
        self._logits = logits
        
    def clear_logits(self):
        self._logits = None
        
    def clip_gradients(self, min: float = -1e1, max: float = 1e1) -> None:
        self._clip_gradients = (min, max)

    def has_gradients_clipped(self) -> bool:
        return self._clip_gradients is not None

    def get_clip_gradients(self) -> None|tuple[float, float]:
        return self._clip_gradients
        
    def clone_hyper_params(self, block: "Block"):
        self.mode = block.mode
        self.padding_type = block.padding_type
        self.regularization_lambda = block.regularization_lambda
        self.global_optimizer = block.global_optimizer
        self.loss_function = block.loss_function