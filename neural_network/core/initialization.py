from abc import ABC, abstractmethod
from typing import Union
from neural_network.gcpu import gcpu

class Initialization(ABC):
    @staticmethod
    def variance() -> Union[int, float]:
        return 1
    
    @abstractmethod
    def generate_layer(self, input_size: int, size: int) -> list:
        pass

    def generate_layer_bias(self, size: int) -> list[gcpu.ndarray]:
        return gcpu.zeros(size) * 0.1
    
    def kernel_filters(self, filter_number: int, filter_shape: tuple[int, int], channels_number: int) -> gcpu.ndarray:
        generator = gcpu.random
        fan_in = channels_number * filter_shape[0] * filter_shape[1]
        scale = gcpu.sqrt(2.0 / fan_in) 
        return generator.normal(
            0, scale,
            (filter_number, channels_number, filter_shape[0], filter_shape[1])
        )
    
    def kernel_bias(self, number: int) -> gcpu.ndarray:
        return gcpu.zeros(number) * 0.1