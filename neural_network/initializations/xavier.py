from neural_network.core import Initialization
import numpy as np

class Xavier(Initialization):
    
    def generate_filters(self, input_shape: tuple, num_filters: int, filter_size: tuple) -> np.ndarray:
    
        filter_height, filter_width = filter_size
        _, _, input_channels = input_shape  # Descompactando input_shape
        
        # A fórmula Xavier para inicialização dos filtros:
        stddev = np.sqrt(2 / (filter_height * filter_width * input_channels))
        
        # Inicialização normal com a distribuição de Xavier
        filters = np.random.normal(0, stddev, (filter_height, filter_width, input_channels, num_filters))
        
        return filters
    
    def generate_layers(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int, 
        layers_number: int
    ) -> list:
        generator = np.random.default_rng(42) 
        weights = []

        # Inicializa as camadas densas com a inicialização Xavier
        for _ in range(layers_number - 1):
            stddev = np.sqrt(2 / (input_size + hidden_size))
            weights.append(generator.normal(0, stddev, (input_size, hidden_size)))
            input_size = hidden_size

        stddev = np.sqrt(2 / (hidden_size + output_size))
        weights.append(generator.normal(0, stddev, (hidden_size, output_size)))

        return weights
    
    def generate_filters(self, channels_number: int, filter_size: tuple[int, int], num_filters: int):
        
        filter_height, filter_width = filter_size
        fan_in = channels_number * filter_height * filter_width
    
        return np.random.randn(num_filters, channels_number, filter_height, filter_width) * np.sqrt(1.0 / fan_in)
