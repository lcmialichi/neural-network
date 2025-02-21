from abc import ABC, abstractmethod
from typing import Union
from typing import TypeAlias
from neural_network.gcpu import driver

class Initialization(ABC):
    @staticmethod
    def variance() -> Union[int, float]:
        return 1
    
    @abstractmethod
    def generate_layer(self, input_size: int, size: int) -> list:
        pass

    def generate_layer_bias(self, size: int):
        return driver.gcpu.zeros(size, dtype=driver.gcpu.float64)
    
    def kernel_filters(self, filter_number: int, filter_shape: tuple[int, int], channels_number: int):
        generator = driver.gcpu.random
        fan_in = channels_number * filter_shape[0] * filter_shape[1]
        scale = driver.gcpu.sqrt(2.0 / fan_in) 
        return generator.normal(
            0, scale,
            (filter_number, channels_number, filter_shape[0], filter_shape[1])
        )
    
    def kernel_bias(self, number: int):
        return driver.gcpu.zeros(number, dtype=driver.gcpu.float64)