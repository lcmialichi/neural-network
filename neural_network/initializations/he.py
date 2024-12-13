from neural_network.core import Initialization
import numpy as np

class He(Initialization):

    def generate_layers(self, input_size, output_size, hidden_size, layers_number):
        generator = np.random.default_rng(42)
        weights = []
        for _ in range(layers_number - 1):
            stddev = np.sqrt(2 / input_size)
            weights.append(generator.normal(0, stddev, (input_size, hidden_size)))
            input_size = hidden_size
        stddev = np.sqrt(2 / hidden_size)
        weights.append(generator.normal(0, stddev, (hidden_size, output_size)))
        return weights
    
    

    def generate_filters(self, channels_number: int, filter_size: tuple[int, int], num_filters: int):
        
        filter_height, filter_width = filter_size
        fan_in = channels_number * filter_height * filter_width
    
        return np.random.randn(num_filters, channels_number, filter_height, filter_width) * np.sqrt(2.0 / fan_in)