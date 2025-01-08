from abc import ABC
from typing import Union, List
import numpy as np
import pickle
import os

class Initialization(ABC):
    def __init__(self, path: Union[str, None] = None):
        self.data = {}
        self._path = path
        if path:
            try:
                with open(path, 'rb') as f:
                    self.data = pickle.load(f)
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                self.data = {}

    def remove_cache(self) -> None:
        if self._path is not None:
            os.remove(self._path)

    def clear_cached_data(self) -> None:
        if self._path is not None:
            self._path = None
            self.data = {}
            
    @staticmethod
    def variance() -> Union[int, float]:
        return 1

    def generate_bias(self, layers: List[dict], output_size: int) -> List[np.ndarray]:
        if 'bias' in self.data:
            return self.data['bias']
        
        biases = [np.zeros(hidden_layer['size']) for hidden_layer in layers]
        biases.append(np.zeros(output_size))
        return biases
    
    def generate_kernel_bias(self, filters: List[np.ndarray]) -> List[np.ndarray]:
        if 'kernel_bias' in self.data:
            return self.data['kernel_bias']
        
        biases = [np.zeros(f.shape[0]) for f in filters]
        return biases
    
    def get_filters_options(self, filters_options: List[np.ndarray]) -> List[np.ndarray]:
        if 'filters_options' in self.data:
            return self.data['filters_options']
        
        return filters_options
    
    def generate_filters(self, filters_list: List[dict], input_channels: int) -> List[np.ndarray]:
        if 'filters' in self.data:
            return self.data['filters']

        generator = np.random.default_rng(42)
        filters = []

        for filter_config in filters_list:
            filter_number = filter_config['number']
            filter_shape = filter_config['shape']
            filter_height, filter_width = filter_shape
            fan_in = input_channels * filter_height * filter_width

            filters.append(generator.normal(
                0, np.sqrt(2.0 / fan_in),
                (filter_number, input_channels, filter_height, filter_width)
            ))

            input_channels = filter_number

        return filters

    def store(
        self, 
        bias: List[np.ndarray] = [], 
        filters: List[np.ndarray] = [], 
        layers: List[np.ndarray] = [], 
        filters_options: List[np.ndarray] = []
    ) -> None:
        if not self._path:
            raise ValueError("Path not specified for storing data.")
        
        directory = os.path.dirname(self._path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        data_to_store = {}
        if bias:
            data_to_store['bias'] = bias
        if filters:
            data_to_store['filters'] = filters
        if filters_options:
            data_to_store['filters_options'] = filters_options
        if layers:
            data_to_store['layers'] = layers
            
        try:
            with open(self._path, 'wb') as f:
                pickle.dump(data_to_store, f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass
            
    
    def save_data(self) -> bool:
        return self._path is not None
