from typing import Callable
from neural_network.core.padding import Padding
from neural_network.core import Initialization
from neural_network.core import Activation

class CnnConfiguration:
    def __init__(self):
        self.config: dict = {}
        self.config['hidden_layers'] = []
        self.config['filters'] = []
    
    def get_config(self) -> dict:
        return self.config
        
    def input_shape(self, channels: int, height: int, width: int) -> "CnnConfiguration":
        self.config['input_shape'] = (channels, height, width)
        return self
    
    def add_hidden_layer(self, size: int, resolution: Callable = None) -> "CnnConfiguration":
        self.config['hidden_layers'].append({
            'size': size,
            'resolution': resolution
        })
        return self
        
    def output_size(self, size: int) -> "CnnConfiguration":
        self.config['output_size'] = size
        return self
    
    def learning_rate(self, rate: float) -> "CnnConfiguration":
        self.config['learning_rate'] = rate
        return self
        
    def regularization_lambda(self, regularization: float) -> "CnnConfiguration":
        self.config['regularization_lambda'] = regularization
        return self
    
    def dropout_rate(self, rate: float) -> "CnnConfiguration":
        self.config['dropout_rate'] = rate
        return self
    
    def stride(self, stride: int) -> "CnnConfiguration":
        self.config['stride'] = stride
        return self
         
    def padding_type(self, padding: Padding) -> "CnnConfiguration":
        self.config['padding_type'] = padding
        return self

    def add_filter(self, filter_number: int, filter_shape: tuple[int, int] = (3, 3)) -> "CnnConfiguration":
        self.config['filters'].append({
            'number': filter_number,
            'shape': filter_shape
        })
        return self
    
    def enable_optimazer(self) -> "CnnConfiguration":
        self.config['optimize'] = True
        return self
    
    def disable_optimazer(self) -> "CnnConfiguration":
        self.config['optimize'] = False
        return self
    
    def with_initializer(self, initializer: Initialization)-> "CnnConfiguration":
        self.config['initializer'] = initializer
        return self
    
    def with_activation(self, activation: Activation)-> "CnnConfiguration":
        self.config['activation'] = activation
        return self
    
    