from Initialization import Initialization
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