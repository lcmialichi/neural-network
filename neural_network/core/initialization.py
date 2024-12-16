from abc import ABC
from typing import Union
import numpy as np

class Initialization(ABC):
    
    @staticmethod
    def variance() -> Union[int, float]:
        return 1
    
    def generate_layers(self, hidden_layers: list, input_size: int, output_size: int) -> list:
        ...
    
    def generate_bias(self, layers: list, output_size: int) -> list:
            biases = [np.zeros(hidden_layer['size']) for hidden_layer in layers]
            biases.append(np.zeros(output_size))
            return biases
    
    def generate_filters(self, filters_list: list, input_channels: int) -> np.ndarray:
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
    
