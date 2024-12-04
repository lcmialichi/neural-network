from neural_network.core import Initialization
import numpy as np

class Xavier(Initialization):
       
    def generate_layers(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int, 
        layers_number: int
    ) -> list:
        generator = np.random.default_rng(42) 
        weights = []

        for _ in range(layers_number - 1):
            stddev = np.sqrt(2 / (input_size + hidden_size))
            weights.append(generator.normal(0, stddev, (input_size, hidden_size)))
            input_size = hidden_size

        stddev = np.sqrt(2 / (hidden_size + output_size))
        weights.append(generator.normal(0, stddev, (hidden_size, output_size)))

        return weights