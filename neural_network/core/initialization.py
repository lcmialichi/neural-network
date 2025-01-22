from abc import ABC
from typing import Union, List
from neural_network.gcpu import gcpu
import pickle
import os

class Initialization(ABC):
            
    @staticmethod
    def variance() -> Union[int, float]:
        return 1

    def generate_bias(self, layers: List[dict], output_size: int) -> List[gcpu.ndarray]:
        biases = [gcpu.zeros(hidden_layer['size']) for hidden_layer in layers]
        biases.append(gcpu.zeros(output_size))
        return biases
    
    def generate_kernel_bias(self, filters: List[gcpu.ndarray]) -> List[gcpu.ndarray]:
        biases = [gcpu.zeros(f.shape[0]) for f in filters]
        return biases
    
    def get_filters_options(self, filters_options: List[gcpu.ndarray]) -> List[gcpu.ndarray]:
        return filters_options
    
    def generate_filters(self, filters_list: List[dict], input_channels: int) -> List[gcpu.ndarray]:
        generator = gcpu.random.default_rng(42)
        filters = []

        for filter_config in filters_list:
            filter_number = filter_config['number']
            filter_shape = filter_config['shape']
            filter_height, filter_width = filter_shape
            fan_in = input_channels * filter_height * filter_width

            filters.append(generator.normal(
                0, gcpu.sqrt(2.0 / fan_in),
                (filter_number, input_channels, filter_height, filter_width)
            ))

            input_channels = filter_number

        return filters

    def kernel_filters(self, filter_number: int, filter_shape: tuple[int, int], channels_number: int) -> gcpu.ndarray:
        generator = gcpu.random.default_rng(42)
        fan_in = channels_number * filter_shape[0] * filter_shape[1]
        return generator.normal(
            0, gcpu.sqrt(2.0 / fan_in),
            (filter_number, channels_number, filter_shape[0],  filter_shape[1])
        )
    
    def kernel_bias(self, number: int) -> gcpu.ndarray:
        return gcpu.zeros(number)