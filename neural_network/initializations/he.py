from neural_network.core import Initialization
import numpy as np

class He(Initialization):

    def generate_layers(self, hidden_layers: list, input_size: int, output_size: int) -> list:
        generator = np.random.default_rng(42)
        weights = []

        for hidden_layer in hidden_layers:
            stddev = np.sqrt(2 / input_size)
            weights.append(generator.normal(0, stddev, (input_size, hidden_layer['size'])))
            input_size = hidden_layer['size']

        stddev = np.sqrt(2 / input_size)
        weights.append(generator.normal(0, stddev, (input_size, output_size)))
        return weights
