from abc import ABC
from typing import Union, List
import numpy as np
import pickle
import os

class Initialization(ABC):
            
    @staticmethod
    def variance() -> Union[int, float]:
        return 1

    def generate_bias(self, layers: List[dict], output_size: int) -> List[np.ndarray]:
        biases = [np.zeros(hidden_layer['size']) for hidden_layer in layers]
        biases.append(np.zeros(output_size))
        return biases
    
    def generate_kernel_bias(self, filters: List[np.ndarray]) -> List[np.ndarray]:
        biases = [np.zeros(f.shape[0]) for f in filters]
        return biases
    
    def get_filters_options(self, filters_options: List[np.ndarray]) -> List[np.ndarray]:
        return filters_options
    
    def generate_filters(self, filters_list: List[dict], input_channels: int) -> List[np.ndarray]:
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
