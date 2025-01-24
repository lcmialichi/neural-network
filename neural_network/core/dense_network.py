from neural_network.gcpu import gcpu
from neural_network.core import Activation
from typing import Union
from neural_network.core import Initialization
from neural_network.initializations import Xavier
from neural_network.core.base_network import BaseNetwork
from neural_network.train import DenseTrainer
from neural_network.optimizers import Adam
from neural_network.foundation import HiddenLayer
from neural_network.foundation import Output

class DenseNetwork(BaseNetwork):

    hidden_outputs: list = []

    def __init__(self, config: dict):
        self._global_optimizer = config.get('global_optimizer', Adam(learning_rate=0.01))
        self._output: Output = config.get('output')
        self.regularization_lambda: float = config.get('regularization_lambda', 0.01)
        self._hidden_layers: list[HiddenLayer] = config.get('hidden_layers', [])
        self.loss_function = config.get('loss_function')
        input_size = config.get('input_size', 0)
        for layer in self._hidden_layers:
            layer.initialize(input_size)
            input_size = layer.size

        self._output.initialize(input_size)

    def forward(self, x: gcpu.ndarray) -> gcpu.ndarray:
        self.hidden_outputs = []
        output = x

        for layer in self._hidden_layers:
            output = gcpu.dot(output, layer.weights()) + layer.bias()
            if layer.has_activation():
                output = layer.get_activation().activate(output)

            if self.is_trainning() and layer.has_dropout():
                output = self._apply_dropout(output, layer.get_dropout())

            self.hidden_outputs.append(output)

        output = gcpu.dot(output, self._output.weights()) + self._output.bias()
        if self._output.has_activation():
            output = self._output.get_activation().activate(output)
        
        return output

    def backward(self, x: gcpu.ndarray, y: gcpu.ndarray, output: gcpu.ndarray):
        if not self._output.has_loss_function():
            raise RuntimeError('ouput loss function not defined')
            
        deltas = [self._output.get_loss_function().gradient(output, y)]

        layer_error = deltas[-1].dot(self._output.weights().T)
        if self._output.has_activation():
            layer_error *= self._output.get_activation().derivate(self.hidden_outputs[-1])

        deltas.append(layer_error)

        for i in range(len(self._hidden_layers) - 1, 0, -1):
            layer = self._hidden_layers[i]
            layer_error = deltas[-1].dot(layer.weights().T)
            if layer.has_activation():
                layer_error *= layer.get_activation().derivate(self.hidden_outputs[i - 1])

            deltas.append(layer_error)

        deltas.reverse()

        for i, layer in enumerate(self._hidden_layers):
            optimizer = layer.get_optimizer() or self._global_optimizer

            input_activation = x if i == 0 else self.hidden_outputs[i - 1]
            grad_weight = input_activation.T.dot(deltas[i]) + self.regularization_lambda * layer.weights()
            layer.update_weights(optimizer.update(f"weights_{i}", layer.weights(), grad_weight))

            grad_bias = gcpu.sum(deltas[i], axis=0, keepdims=True)
            layer.update_bias(optimizer.update(f"biases_{i}", layer.bias(), grad_bias))

        optimizer = self._output.get_optimizer() or self._global_optimizer
        grad_weight = self.hidden_outputs[-1].T.dot(deltas[-1]) + self.regularization_lambda * self._output.weights()
        self._output.update_weights(optimizer.update("weights_output", self._output.weights(), grad_weight))

        grad_bias = gcpu.sum(deltas[-1], axis=0, keepdims=True)
        self._output.update_bias(optimizer.update("biases_output", self._output.bias(), grad_bias))

        return deltas[0].dot(self._hidden_layers[0].weights().T).reshape(x.shape)

    def train(self, x_batch: gcpu.ndarray, y_batch: gcpu.ndarray) -> gcpu.ndarray:
        output_batch = self.forward(x_batch)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def predict(self, x: Union[gcpu.ndarray, gcpu.ndarray]) -> gcpu.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    def get_trainer(self):
        
        return DenseTrainer(self)