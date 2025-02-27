from neural_network.core import Initialization
from neural_network.gcpu import driver

class Xavier(Initialization):
    
    def generate_layer(self, input_size: int, size: int) -> list:
        generator = driver.gcpu.random
        stddev = driver.gcpu.sqrt(2 / (input_size + size))
        return generator.normal(0, stddev, (input_size, size))

    def kernel_filters(self, filter_number: int, filter_shape: tuple[int, int], channels_number: int):
        generator = driver.gcpu.random
        fan_in = channels_number * filter_shape[0] * filter_shape[1]
        fan_out = filter_number * filter_shape[0] * filter_shape[1]
        scale = driver.gcpu.sqrt(2.0 / (fan_in + fan_out))
        return generator.normal(0, scale, (*filter_shape, channels_number, filter_number))
        
    def generate_layer_bias(self, size: int) -> list:
        return driver.gcpu.zeros(size)
    
    def kernel_bias(self, number: int):
        return driver.gcpu.zeros(number)
