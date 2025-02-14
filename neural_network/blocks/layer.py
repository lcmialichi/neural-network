from neural_network.core import Activation
from neural_network.core.optimizer import Optimizer
from neural_network.core import Initialization
from neural_network.core.dropout import Dropout
from neural_network.gcpu import driver
import uuid
import neural_network.supply as attr
from .block import Block

class Layer(Block):
    def __init__(
        self,
        size: int, 
        dropout: float | None = None,
        ):
        self._bias: list | None = None
        self._weights: list | None = None
        self.size = size
        self._dropout = Dropout(dropout) if dropout else None
        self.layer_id = str(uuid.uuid4())
    
    def bias(self):
        return self._bias
    
    def update_bias(self, bias):
        self._bias = bias
    
    def weights(self):
        return self._weights
    
    def update_weights(self, weights):
        self._weights = weights
        
    def boot(self, shape: tuple):
        self._weights = self._initializer.generate_layer(shape[1], self.size)
        self._bias = self._initializer.generate_layer_bias(self.size)

    def forward(self, input):
        assert input.shape[1] == self.weights().shape[0], "Dimensão incompatível entre input e pesos"
        
        if self._weights is None and self._bias is None:
            self.boot(input.shape)

        self.clear_logits()
        output = driver.gcpu.dot(input, self.weights()) + self.bias()
        self.store_logits(output)

        if self.has_activation():
            output = self.get_activation().activate(output)
                
        if self.mode == 'train' and self.has_dropout():
            output = self.get_dropout().forward(output)
            
        return output
    
    
    def backward(self, input_layer, y, delta):
        if self.mode == 'train' and self.has_dropout():
            delta = self.get_dropout().backwards(delta)
        
        if self.has_activation() and not (isinstance(self.get_activation(), attr.Softmax) and isinstance(self.loss_function, attr.CrossEntropyLoss)):
            delta *= self.get_activation().derivate(self.logits())

        grad_weight = input_layer.T.dot(delta)
        grad_bias = driver.gcpu.sum(delta, axis=0, keepdims=True)

        if self.has_gradients_clipped():
            min_c, max_c = self.get_clip_gradients()
            grad_weight = driver.gcpu.clip(grad_weight, min_c, max_c)
            grad_bias = driver.gcpu.clip(grad_bias, min_c, max_c)

        self.update_weights(self.get_optimizer().update(f"weights_{self.layer_id}", self.weights(), grad_weight))
        self.update_bias(self.get_optimizer().update(f"biases_{self.layer_id}", self.bias(), grad_bias, weight_decay=False))

        return delta.dot(self.weights().T)
