from neural_network.loss import LossFunction
from neural_network.core.cnn_network import CnnNetwork
from neural_network.core import Activation
from neural_network.activations import Softmax
from neural_network.core.processor import Processor
from neural_network.storage import Storage
from neural_network.core.optimizer import Optimizer
from neural_network.foundation import Kernel
from neural_network.foundation import Output
from neural_network.foundation import HiddenLayer
from typing import Union
from neural_network.core.padding import Padding

class CnnConfiguration:
    def __init__(self, config: dict = None, storage: Union[None, Storage] = None):
        if config is None:
            config = {}
        
        self._storage = storage
        self._config: dict = config
        self._config['hidden_layers'] = self._config.get('hidden_layers', [])
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
        self, 
        size: int, 
        dropout: Union[float, None] = None,
        ) -> "HiddenLayer":

        layer = HiddenLayer(size=size, dropout=dropout)
        self._config['hidden_layers'].append(layer)
        return layer
    
    def output(self, size: int) -> "Output":
        output = Output(size=size)
        self._config['output'] = output
        return output
        
    def output_size(self, size: int) -> "CnnConfiguration":
        self._config['output_size'] = size
        return self
    
    def learning_rate(self, rate: float) -> "CnnConfiguration":
        self._config['learning_rate'] = rate
        return self
        
    def regularization_lambda(self, regularization: float) -> "CnnConfiguration":
        self._config['regularization_lambda'] = regularization
        return self
    
    def add_kernel(self, number: int, shape: tuple[int, int] = (3, 3), stride: int = 1) -> "Kernel":
        kernel = Kernel(number=number, shape=shape, stride=stride)
        self._config['kernels'].append(kernel)
        return kernel
    
    def padding_type(self, padding: Padding) -> "CnnConfiguration":
        self._config['padding_type'] = padding
        return self

    def restore_initialization_cache(self):
        if self._storage and self._storage.has():
            self._storage.remove()
            
    def with_no_cache(self):
        self._storage = None
    
    def set_processor(self, processor: Processor):
        self._config['processor'] = processor
    
    
    