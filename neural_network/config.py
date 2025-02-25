from neural_network.supply.loss import LossFunction
from neural_network.core.model import Model
from neural_network.blocks import Flatten, Dense, GlobalAveragePooling, Kernel, Propagable
from neural_network.core.processor import Processor
from neural_network.storage import Storage
from neural_network.core.optimizer import Optimizer
from typing import Union
from neural_network.core.padding import Padding
from neural_network.configuration import Driver, GlobalConfig

class Config:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        
        self._storage = None
        self._config: dict = config
        self._config['blocks'] = self._config.get('blocks', [])
        GlobalConfig().set_driver(Driver['cpu'])
        
    def driver(self, name: str) -> None:
        GlobalConfig().set_driver(Driver[name])

    def with_cache(self, path: str) -> "Config":
        self._storage = Storage(path)
        return self
    
    def set_global_optimizer(self, optimizer: Optimizer):
        self._config['global_optimizer'] = optimizer
    
    def new_model(self) -> "Model":
        if self._storage and self._storage.has():
            return self._storage.get()
        
        return Model(self.get_config(), self._storage)
    
    def get_config(self) -> dict:
        return self._config
        
    def l2_regularization(self, rate: float) -> "Config":
        self._config['regularization_lambda'] = rate
        return self
    
    def add_kernel(self, number: int, shape: tuple[int, int] = (3, 3), stride: int = 1, bias: bool = True) -> "Kernel":
        kernel = Kernel(number=number, shape=shape, stride=stride, bias=bias)
        self._add_block(kernel)
        return kernel
    
    def add(self, custom: Propagable):
        self._add_block(custom)
    
    def flatten(self):
        flatten = Flatten()
        self._add_block(flatten)
        return flatten
    
    def dense(self):
        dense = Dense()
        self._add_block(dense)
        return dense
    
    def global_avg_pooling(self) -> "GlobalAveragePooling":
        avg_pooling = GlobalAveragePooling()
        self._add_block(avg_pooling)
        return avg_pooling
    
    def custom(self, custom):
        self._add_block(custom)
    
    def padding_type(self, padding: Padding) -> "Config":
        self._config['padding_type'] = padding
        return self

    def restore_initialization_cache(self):
        if self._storage and self._storage.has():
            self._storage.remove()
            
    def with_no_cache(self):
        self._storage = None
    
    def loss_function(self, loss_function: LossFunction):
        self._config['loss_function'] = loss_function

    def _add_block(self, block):
        self._config['blocks'].append(block)

    
    
    