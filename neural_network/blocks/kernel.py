from neural_network.core import Activation, Initialization
from neural_network.core.dropout import Dropout
from neural_network.supply import initializations, normalization, pooling
from neural_network.core.pooling import Pooling
from neural_network.core.optimizer import Optimizer
from neural_network.support import conv, im2col, get_padding, add_padding
from neural_network.gcpu import driver
from .block import Block

import uuid

class Kernel(Block):
    
    _filters: list | None = None
    _bias: list | None = None
    _dropout: Dropout | None = None
    _initializer: Initialization = initializations.XavierUniform()
    _activation: Activation | None = None
    _batch_normalization: normalization.BatchNormalization | None  = None
    _pooling: Pooling | None = None
    _optimizer: Optimizer | None = None
    _clip_gradients: tuple[float, float] | None = None
    
    def __init__(self, number: int, shape: tuple[int, int] = (3, 3), stride: int = 1, bias: bool = True):
        self.number = number
        self.shape = shape
        self.stride = stride
        self._apply_bias = bias
        self.kernel_id = str(uuid.uuid4())

    def has_pooling(self) -> bool:
        return self._pooling is not None
    
    def get_pooling(self) -> "Pooling":
        return self._pooling

    def max_pooling(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self._pooling = pooling.MaxPooling(shape=shape, stride=stride)
    
    def avg_pooling(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self._pooling = pooling.AvgPooling(shape=shape, stride=stride)

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

    def boot(self, shape: tuple):
        self._filters = self._initializer.kernel_filters(self.number, self.shape, shape[1])
        self._bias = self._initializer.kernel_bias(self.number)
    
    def forward(self, x):
        if self._filters is None and self._bias is None:
            self.boot(x.shape)

        self.clear_logits()
        logit = conv(x, self.filters(), self.number, self.stride, self.shape, self.padding_type)
        if self._apply_bias:
            logit += self.bias()[:, driver.gcpu.newaxis, driver.gcpu.newaxis]
       
        conv_output = logit
        if self.has_activation():
            conv_output = self.get_activation().activate(conv_output)

        if self.has_batch_normalization():
            conv_output = self.get_batch_normalization().batch_normalize(
                x=conv_output, mode=self.mode
            )

        if self.has_pooling():
            conv_output = self.get_pooling().apply_pooling(conv_output)
            
        if self.mode == 'train' and self.has_dropout():
            conv_output = self.get_dropout().forward(conv_output)
        
        self.store_logits(logit)
        return conv_output
    
    def backward(self, input_layer, y, delta):
        filters = self.filters()
        num_filters, input_channels, fh, fw = filters.shape
        if self.mode == 'train' and self.has_dropout():
            delta = self.get_dropout().backwards(delta)
            
        if self.has_pooling():
            delta = self.get_pooling().unpooling(delta)

        if self.has_batch_normalization():
            bn = self.get_batch_normalization()
            delta, dgamma, dbeta = bn.batch_norm_backward(delta)
            bn.update_gama(self.get_optimizer().update(f'bn_gamma_{self.kernel_id}', bn.get_gama(), dgamma, weight_decay=False))
            bn.update_beta(self.get_optimizer().update(f'bn_beta_{self.kernel_id}', bn.get_beta(), dbeta, weight_decay=False))

        if self.has_activation():
            delta *= self.get_activation().derivate(self.logits())

        padding = get_padding(
            (input_layer.shape[2], input_layer.shape[3]), (fh, fw), self.stride, self.padding_type
        )         

        grad_bias = driver.gcpu.sum(delta, axis=(0, 2, 3))
        batch_size, _, output_h, output_w = delta.shape
        input_reshaped = im2col(add_padding(input_layer, padding), (fh, fw), self.stride)
        delta_reshaped = delta.reshape(batch_size * output_h * output_w, num_filters)
        grad_filter = driver.gcpu.matmul(delta_reshaped.T, input_reshaped).reshape(filters.shape)
        
        if self.has_gradients_clipped():
            min_c, max_c = self.get_clip_gradients()
            grad_filter = driver.gcpu.clip(grad_filter, min_c, max_c)
            grad_bias = driver.gcpu.clip(grad_bias, min_c, max_c)
        
        if self._apply_bias:
            self.update_bias(self.get_optimizer().update(f"kernel_bias_{self.kernel_id}", self.bias(), grad_bias, weight_decay=False))
            
        self.update_filters(self.get_optimizer().update(f"kernel_filters_{self.kernel_id}", self.filters(), grad_filter))

        flipped_filters = driver.gcpu.flip(filters, axis=(2, 3))
        delta_col = driver.gcpu.matmul(
            delta.reshape(batch_size * output_h * output_w, num_filters), 
            flipped_filters.reshape(num_filters, -1)
        )
        
        delta = delta_col.reshape(batch_size, output_h, output_w, input_channels, fh, fw)
        delta = delta.transpose(0, 3, 1, 2, 4, 5).sum(axis=(4, 5))

        if self.stride > 1:
            expanded_delta = driver.gcpu.zeros(
                (batch_size, num_filters, output_h * self.stride, output_w * self.stride)
            )
            expanded_delta[:, :, ::self.stride, ::self.stride] = delta
            delta = expanded_delta

        return delta