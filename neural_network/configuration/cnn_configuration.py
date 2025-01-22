from neural_network.core.padding import Padding
from neural_network.core import Initialization
from neural_network.core.cnn_network import CnnNetwork
from neural_network.core import Activation
from neural_network.activations import Relu
from neural_network.activations import Softmax
from neural_network.core.processor import Processor
from neural_network.storage import Storage
from neural_network.core.optimizer import Optimizer
from neural_network.foundation.kernel import Kernel
from typing import Union
from neural_network.gcpu import gcpu

class CnnConfiguration:
    def __init__(self, config: dict = None, storage: Union[None, Storage] = None):
        if config is None:
            config = {}
        
        self._storage = storage
        self._config: dict = config
        self._config['hidden_layers'] = self._config.get('hidden_layers', [])
        self._config['filters'] = self._config.get('filters', [])
        self._config['kernels'] = self._config.get('kernels', [])
    
    def with_cache(self, path: str) -> "CnnConfiguration":
        self._storage = Storage(path)
        return self
    
    def set_global_optimizer(self, optimizer: Optimizer):
        self._config['global_optimizer'] = optimizer
    
    def new_model(self) -> "CnnNetwork":
        if self._storage and self._storage.has():
            return self._storage.get()
        
        return CnnNetwork(self.get_config(), self._storage)
    
    def get_config(self) -> dict:
        return self._config
        
    def input_shape(self, channels: int, height: int, width: int) -> "CnnConfiguration":
        self._config['input_shape'] = (channels, height, width)
        return self
    
    def add_hidden_layer(
        self, size: int, 
        activation: Activation = Relu(),
        dropout: Union[float, None] = None,
        optimizer: Union[Optimizer, None] = None
        ) -> "CnnConfiguration":
        self._config['hidden_layers'].append({
            'size': size,
            'activation': activation,
            'dropout': dropout,
            'optimizer': optimizer
        })
        return self
    
    def output(self, size: int, activation: Activation = Softmax()):
        self._config['output'] = {
            'size': size,
            'activation': activation
        }
        return self
        
    def output_size(self, size: int) -> "CnnConfiguration":
        self._config['output_size'] = size
        return self
    
    def learning_rate(self, rate: float) -> "CnnConfiguration":
        self._config['learning_rate'] = rate
        return self
        
    def regularization_lambda(self, regularization: float) -> "CnnConfiguration":
        self._config['regularization_lambda'] = regularization
        return self
    
    def padding_type(self, padding: Padding) -> "CnnConfiguration":
        self._config['padding_type'] = padding
        return self

    def add_filter(
        self, 
        filter_number: int, 
        filter_shape: tuple[int, int] = (3, 3),
        stride: int = 1,
        activation: Activation = Relu(),
        dropout: Union[float, None] = None,
        optimizer: Union[Optimizer, None] = None
    ) -> "CnnConfiguration":
        self._config['filters'].append({
            'number': filter_number,
            'shape': filter_shape,
            'stride': stride,
            'activation': activation,
            'dropout': dropout,
            'optimizer': optimizer
        })

        return self
    
    def add_kernel(self, number: int, shape: tuple[int, int] = (3, 3), stride: int = 1) -> "Kernel":
        assert stride == 1, "not working with stride > 1 yet"
        kernel = Kernel(number=number, shape=shape, stride=stride)
        self._config['kernels'].append(kernel)
        return kernel
    
    def add_batch_normalization(self, gama: float = 1.0, beta: float = 0.0, momentum: float = 0.9 )-> "CnnConfiguration":
        assert len(self._config['filters']) > 0
        assert momentum > 0 and momentum < 1, "Momentum must be between 0 and 1"
        
        num_filters =  self._config['filters'][-1]['number']
        
        self._config['filters'][-1]['bn'] = {
            'gamma': gcpu.ones((1, num_filters, 1, 1)) * gama,
            'beta': gcpu.zeros((1, num_filters, 1, 1)) + beta,
            'running_mean': gcpu.zeros((1, num_filters, 1, 1)),
            'running_var': gcpu.ones((1, num_filters, 1, 1)),
            'momentum': momentum 
        }
        
        return self
    
    def add_polling(self, polling_shape: tuple[int, int] = (2, 2), stride: int = 1):
        assert len(self._config['filters']) > 0
        
        self._config['filters'][-1]['polling'] = {
            'shape': polling_shape,
            'stride': stride
        }
    
    def enable_optimazer(self) -> "CnnConfiguration":
        self._config['optimize'] = True
        return self
    
    def disable_optimazer(self) -> "CnnConfiguration":
        self._config['optimize'] = False
        return self
    
    def with_initializer(self, initializer: Initialization)-> "CnnConfiguration":
        self._config['initializer'] = initializer
        return self
    
    def with_activation(self, activation: Activation)-> "CnnConfiguration":
        self._config['activation'] = activation
        return self
    
    def restore_initialization_cache(self):
        if self._storage and self._storage.has():
            self._storage.remove()
            
    def with_no_cache(self):
        if self._storage:
            self._storage = None
    
    def set_processor(self, processor: Processor):
        self._config['processor'] = processor
    
    
    