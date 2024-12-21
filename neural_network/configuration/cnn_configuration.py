from neural_network.core.padding import Padding
from neural_network.core import Initialization
from neural_network.core.cnn_network import CnnNetwork
from neural_network.core import Activation
from neural_network.activations import Relu

class CnnConfiguration:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
            
        self._config: dict = config
        self._config['hidden_layers'] = self._config.get('hidden_layers', [])
        self._config['filters'] = self._config.get('filters', [])
    
    def new_model(self) -> "CnnNetwork":
        return CnnNetwork(self.get_config())
    
    def get_config(self) -> dict:
        return self._config
        
    def input_shape(self, channels: int, height: int, width: int) -> "CnnConfiguration":
        self._config['input_shape'] = (channels, height, width)
        return self
    
    def add_hidden_layer(self, size: int, activation: Activation = Relu()) -> "CnnConfiguration":
        self._config['hidden_layers'].append({
            'size': size,
            'activation': activation
        })
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
    
    def dropout_rate(self, rate: float) -> "CnnConfiguration":
        self._config['dropout_rate'] = rate
        return self
    
    def stride(self, stride: int) -> "CnnConfiguration":
        self._config['stride'] = stride
        return self
         
    def padding_type(self, padding: Padding) -> "CnnConfiguration":
        self._config['padding_type'] = padding
        return self

    def add_filter(
        self, filter_number: int, 
        filter_shape: tuple[int, int] = (3, 3),
        stride: int = 1,
        activation: Activation = Relu()
    ) -> "CnnConfiguration":
        self._config['filters'].append({
            'number': filter_number,
            'shape': filter_shape,
            'stride': stride,
            'activation': activation
        })

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
    
    
    
    