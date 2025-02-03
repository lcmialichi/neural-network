from neural_network.initializations import He
from neural_network.normalization import BatchNormalization
from neural_network.pooling import MaxPooling, AvgPooling
from neural_network.core import Activation, Initialization
from neural_network.core.dropout import Dropout
from neural_network.core.pooling import Pooling
from neural_network.core.optimizer import Optimizer
from typing import Callable
from neural_network.support import conv, im2col, get_padding, add_padding
from neural_network.gcpu import gcpu
import uuid

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
    _logits = None
    mode = None
    padding_type = None
    regularization_lambda = None
    
    def __init__(self, number: int, shape: tuple[int, int] = (3, 3), stride: int = 1):
        self.number = number
        self.shape = shape
        self.stride = stride
        self.kernel_id = str(uuid.uuid4())

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
    
    def clip_gradients(self, min: float = -1e1, max: float = 1e1) -> None:
        self._clip_gradients = (min, max)

    def has_gradients_clipped(self) -> bool:
        return self._clip_gradients is not None

    def get_clip_gradients(self) -> None|tuple[float, float]:
        return self._clip_gradients
    
    def logits(self):
        return self._logits
    
    def conv(self):
        return self._conv
    
    def store_logit(self, logits):
        self._logits = logits
    
    def forward(self, x):
        if not self._filters:
            self._filters = self._initializer.kernel_filters(self.number, self.shape, x.shape[1])
        
        if not self._bias:
            self._bias = self._initializer.kernel_bias(self.number)

        logit = conv(x, self.filters(), self.number, self.stride, self.shape, self.padding_type)
        logit += self.bias()[:, gcpu.newaxis, gcpu.newaxis]
        if self.has_batch_normalization():
            logit = self.get_batch_normalization().batch_normalize(
                x=logit, mode=self.mode
            )
        
        conv_output = logit
        if self.has_activation():
            conv_output = self.get_activation().activate(conv_output)

        if self.has_pooling():
            conv_output = self.get_pooling().apply_pooling(conv_output)
            
        if self.mode == 'train' and self.has_dropout():
            conv_output = self.get_dropout().apply(conv_output)
            
        self.store_logit(logit)
        return conv_output
    
    def backward(
        self, 
        input_layer, 
        delta, 
        ):
        filters = self.filters()
        num_filters, input_channels, fh, fw = filters.shape
        optimizer = self.get_optimizer()
        
        if self.mode and self.has_dropout():
            delta = self.get_dropout().scale_correction(delta)
            
        if self.has_pooling():
            delta = self.get_pooling().unpooling(delta)

        if self.has_activation():
            delta *= self.get_activation().derivate(self.logits())
            
        if self.has_batch_normalization():
            bn = self.get_batch_normalization()
            delta, dgamma, dbeta = bn.batch_norm_backward(delta)
            bn.update_gama(optimizer.update(f'bn_gamma_{self.kernel_id}', bn.get_gama(), dgamma))
            bn.update_beta(optimizer.update(f'bn_beta_{self.kernel_id}', bn.get_beta(), dbeta))

        padding = get_padding(
            (input_layer.shape[2], input_layer.shape[3]), (fh, fw), self.stride, self.padding_type
        )         

        grad_bias = gcpu.sum(delta, axis=(0, 2, 3))
        batch_size, _, output_h, output_w = delta.shape
        input_reshaped = im2col(add_padding(input_layer, padding), (fh, fw), self.stride)
        delta_reshaped = delta.reshape(batch_size * output_h * output_w, num_filters)
        grad_filter = gcpu.matmul(delta_reshaped.T, input_reshaped).reshape(filters.shape)
        grad_filter += self.regularization_lambda * filters
        
        if self.has_gradients_clipped():
            min_c, max_c = self.get_clip_gradients()
            grad_filter = gcpu.clip(grad_filter, min_c, max_c)
            grad_bias = gcpu.clip(grad_bias, min_c, max_c)
            
        self.update_bias(optimizer.update(f"kernel_bias_{self.kernel_id}", self.bias(), grad_bias))
        self.update_filters(optimizer.update(f"kernel_filters_{self.kernel_id}", self.filters(), grad_filter))
        self._sum_gradients = None
        
        delta_col = gcpu.matmul(delta_reshaped, gcpu.flip(filters, axis=(2, 3)).reshape(num_filters, -1))
        delta = delta_col.reshape(batch_size, output_h, output_w, input_channels, fh, fw)
        return delta.transpose(0, 3, 4, 5, 1, 2).sum(axis=(2, 3))