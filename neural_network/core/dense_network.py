import numpy as np
from neural_network.core import Activation
from typing import Union
from neural_network.core import Initialization
from neural_network.initializations import Xavier
from neural_network.core.base_network import BaseNetwork
from neural_network.train import DenseTrainer
from neural_network.optimizers import Adam

class DenseNetwork(BaseNetwork):
    def __init__(self, config: dict, initializer: Initialization = Xavier()):
        self.optimizer = None
        self.input_size: int = config.get('input_size', 0)
        self.hidden_layers: int = config.get('hidden_layers', [])
        self.output_size: int = config.get('output_size', 0)
        self.learning_rate: float = config.get('learning_rate', 0.01)
        self.regularization_lambda: float = config.get('regularization_lambda', 0.01)
        self.dropout_rate: float = config.get('dropout_rate', 0.2)
        self.layers_number: int = len(self.hidden_layers)
    
        self.biases = initializer.generate_bias(
            self.hidden_layers,
            self.output_size
        )
        
        self.weights = initializer.generate_layers(
            self.hidden_layers,
            self.input_size, 
            self.output_size
        )
       
        self.hidden_output: list = []
        self.hidden_activations: list = []
        if config.get('optimize', True):
            self.optimizer = Adam(
                learning_rate=self.learning_rate
                )
   

    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        self.hidden_outputs = []
        output = x
        
        for layer_idx in range(self.layers_number):
            activation: Activation = self.hidden_layers[layer_idx]['activation']
            output = activation.activate(np.dot(output, self.weights[layer_idx]) + self.biases[layer_idx])
            if dropout:
                output = self.apply_dropout(output)
          
            self.hidden_outputs.append(output)

        return self.softmax(np.dot(output, self.weights[-1]) + self.biases[-1])

    def apply_dropout(self, activations: np.ndarray) -> np.ndarray:
        mask = np.random.default_rng(42).integers(0, 2, size=activations.shape)
        activations = activations * mask
        activations /= (1 - self.dropout_rate)
        return activations

    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        output_error = output - y
        deltas = [output_error]

        for i in range(len(self.weights) - 1, 0, -1):
            layer_error = deltas[-1].dot(self.weights[i].T)
            activation: Activation = self.hidden_layers[i -1]['activation']
            layer_delta = layer_error * activation.derivate(self.hidden_outputs[i - 1])
            deltas.append(layer_delta)

        deltas.reverse()
        
        for i in range(len(self.weights)):
            input_activation = x if i == 0 else self.hidden_outputs[i - 1]

            grad_weight = input_activation.T.dot(deltas[i]) + self.regularization_lambda * self.weights[i]
            self.weights[i] = self.optimizer.update(f"weights_{i}", self.weights[i], grad_weight)

            grad_bias = np.sum(deltas[i], axis=0)
            self.biases[i] = self.optimizer.update(f"biases_{i}", self.biases[i], grad_bias)

        return deltas[0].dot(self.weights[0].T).reshape(x.shape)

    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output_batch = self.forward(x_batch, True)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def predict(self, x: Union[np.ndarray, np.ndarray]) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1) 
        return self.forward(x)
    
    def get_trainer(self):
        return DenseTrainer(self)
