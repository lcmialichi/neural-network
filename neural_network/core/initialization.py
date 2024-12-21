from abc import ABC
from typing import Union, List
import numpy as np
import pickle

class Initialization(ABC):
    def __init__(self, path: Union[str, None] = None):
        self._cache: bool = False
        self.data = {}

        if path:
            self._cache = True
            self._path = path
            try:
                with open(path, 'rb') as f:
                    self.data = pickle.load(f)
            except (FileNotFoundError, EOFError, pickle.UnpicklingError):
                self.data = {}

    @staticmethod
    def variance() -> Union[int, float]:
        return 1

    def generate_bias(self, layers: List[dict], output_size: int) -> List[np.ndarray]:
        if self._cache and 'bias' in self.data:
            return self.data['bias']
        
        biases = [np.zeros(hidden_layer['size']) for hidden_layer in layers]
        biases.append(np.zeros(output_size))
        return biases

    def generate_filters(self, filters_list: List[dict], input_channels: int) -> List[np.ndarray]:
        if self._cache and 'filters' in self.data:
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

    def store(self, bias: List[np.ndarray] = [], filters: List[np.ndarray] = [], layers: List[np.ndarray] = []) -> None:
        data_to_store = {}
        if bias:
            data_to_store['bias'] = bias
        if filters:
            data_to_store['filters'] = filters
        if layers:
            data_to_store['layers'] = layers
            
        with open(self._path, 'wb') as f:
            pickle.dump(data_to_store, f)
            
    
    def save_data(self) -> bool:
        return self._cache
