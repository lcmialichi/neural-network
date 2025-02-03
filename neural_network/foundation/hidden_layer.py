from neural_network.core import Activation
from neural_network.core.optimizer import Optimizer
from neural_network.core import Initialization
from neural_network.core.dropout import Dropout
from neural_network.normalization import BatchNormalization
from neural_network.gcpu import gcpu
import uuid

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
        self._clip_gradients: tuple[float, float] | None = None
        self._bias = []
        self._weights = []
        self.size = size
        self._dropout = Dropout(dropout) if dropout else None
        self._logits = None
        self.mode = None
        self.layer_id = str(uuid.uuid4())
        

    def has_dropout(self) -> bool:
        return self._dropout is not None
    
    def dropout(self, rate: float) -> None:
        self._dropout = Dropout(rate)

    def get_dropout(self) -> "Dropout":
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

    def clip_gradients(self, min: float = -1e1, max: float = 1e1) -> None:
        self._clip_gradients = (min, max)

    def has_gradients_clipped(self) -> bool:
        return self._clip_gradients is not None

    def get_clip_gradients(self) -> None|tuple[float, float]:
        return self._clip_gradients
    
    def store_logit(self, logits):
        self._logits = logits
    
    def forward(self, input):
        if not self._weights:
            self._weights = self._initializer.generate_layer(input.shape[1], self.size)
        
        if not self._bias:
            self._bias = self._initializer.generate_layer_bias(self.size)

        output = gcpu.dot(input, self.weights()) + self.bias()
        self.store_logit(output)
        
        if self.has_activation():
            output = self.get_activation().activate(output)
                
        if self.mode == 'train' and self.has_dropout():
            output = self.get_dropout().apply(output)
            
        return output
    
    def backward(
        self,
        input_layer, 
        delta
        ):
        layer_error = delta.dot(self.weights().T)
        
        if self.mode == 'train' and self.has_dropout():
            layer_error = self.get_dropout().scale_correction(layer_error)
            
        if self.has_activation():
            layer_error *= self.get_activation().derivate(self._logits)
        
        grad_weight = input_layer.T.dot(layer_error) + self.regularization_lambda * self.weights()
        grad_bias = gcpu.sum(delta, axis=0, keepdims=True)
        
        if self.has_gradients_clipped():
            min_c, max_c = self.get_clip_gradients()
            grad_weight = gcpu.clip(grad_weight, min_c, max_c)
            grad_bias = gcpu.clip(grad_bias, min_c, max_c)
            
        self.update_weights(self._optimizer.update(f"weights_{self.layer_id}", self.weights(), grad_weight))
        self.update_bias(self._optimizer.update(f"biases_{self.layer_id}", self.bias(), grad_bias))
        
        return layer_error

        