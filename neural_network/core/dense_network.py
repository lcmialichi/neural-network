from neural_network.gcpu import gcpu
from typing import Union
from neural_network.core.base_network import BaseNetwork
from neural_network.train import DenseTrainer
from neural_network.optimizers import Adam
from neural_network.foundation import HiddenLayer
from neural_network.foundation import Output

class DenseNetwork(BaseNetwork):

    hidden_outputs: list = []
    dlogits: list = []

    def __init__(self, config: dict):
        
        self._global_optimizer = config.get('global_optimizer', Adam(learning_rate=0.01))
        self._output: Output = config.get('output')
        self.regularization_lambda: float = config.get('regularization_lambda', 0.001)
        self._hidden_layers: list[HiddenLayer] = config.get('hidden_layers', [])

        input_size = config.get('input_size', 0)
        self._initialize_layers(input_size)
       

    def _initialize_layers(self, input_size: int) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be greater than 0")

        for layer in self._hidden_layers:
            layer.initialize(input_size)
            input_size = layer.size

        self._output.initialize(input_size)


    def forward(self, x: gcpu.ndarray) -> gcpu.ndarray:
        self.hidden_outputs.clear()
        self.dlogits.clear()

        output = x

        for layer in self._hidden_layers:
            output = gcpu.dot(output, layer.weights()) + layer.bias()
            self.dlogits.append(output)
            if layer.has_activation():
                output = layer.get_activation().activate(output)
            
            if self.is_training() and layer.has_dropout():
                output = layer.get_dropout().apply(output)

            self.hidden_outputs.append(output)

        output = gcpu.dot(output, self._output.weights()) + self._output.bias()
        self.dlogits.append(output)

        if self._output.has_activation():
            output = self._output.get_activation().activate(output)
        
        self.hidden_outputs.append(output)
        return output

    def backward(self, x: gcpu.ndarray, y: gcpu.ndarray, output: gcpu.ndarray):
        if not self._output.has_loss_function():
            raise RuntimeError("Output loss function not defined")
        
        deltas = self._compute_deltas(y, output)
        self._update_hidden_layers(x, deltas)
        self._update_output_layer(deltas)
        
        return deltas[0].dot(self._hidden_layers[0].weights().T).reshape(x.shape)

    def _compute_deltas(self, y: gcpu.ndarray, output: gcpu.ndarray) -> list:
        deltas = []
        output_delta = self._output.get_loss_function().gradient(output, y)
        
        if self._output.has_activation():
            output_delta *= self._output.get_activation().derivate(self.dlogits[-1])
        
        deltas.append(output_delta)
        
        for i in range(len(self._hidden_layers) - 1, -1, -1):
            layer = self._hidden_layers[i]
            weights = self._output.weights() if i == len(self._hidden_layers) - 1 else self._hidden_layers[i + 1].weights()
            layer_error = deltas[-1].dot(weights.T)
            
            if self.is_training() and layer.has_dropout():
                layer_error = layer.get_dropout().scale_correction(layer_error)
            
            if layer.has_activation():
                layer_error *= layer.get_activation().derivate(self.dlogits[i])
            
            deltas.append(layer_error)
        
        return deltas[::-1]


    def _update_hidden_layers(self, x: gcpu.ndarray, deltas: list):
        for i, layer in enumerate(self._hidden_layers):
            optimizer = layer.get_optimizer() or self._global_optimizer
            input_activation = x if i == 0 else self.hidden_outputs[i - 1]
            
            grad_weight = input_activation.T.dot(deltas[i]) + self.regularization_lambda * layer.weights()
            grad_bias = gcpu.sum(deltas[i], axis=0, keepdims=True)
            
            if layer.has_gradients_clipped():
                min_c, max_c = layer.get_clip_gradients()
                grad_weight = gcpu.clip(grad_weight, min_c, max_c)
                grad_bias = gcpu.clip(grad_bias, min_c, max_c)
            
            layer.update_weights(optimizer.update(f"weights_{i}", layer.weights(), grad_weight))
            layer.update_bias(optimizer.update(f"biases_{i}", layer.bias(), grad_bias))


    def _update_output_layer(self, deltas: list):
        optimizer = self._output.get_optimizer() or self._global_optimizer
        
        grad_weight = self.hidden_outputs[-2].T.dot(deltas[-1]) + self.regularization_lambda * self._output.weights()
        grad_bias = gcpu.sum(deltas[-1], axis=0, keepdims=True)
        
        self._output.update_weights(optimizer.update("weights_output", self._output.weights(), grad_weight))
        self._output.update_bias(optimizer.update("biases_output", self._output.bias(), grad_bias))


    def train(self, x_batch: gcpu.ndarray, y_batch: gcpu.ndarray) -> gcpu.ndarray:
        output_batch = self.forward(x_batch)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def predict(self, x: Union[gcpu.ndarray, gcpu.ndarray]) -> gcpu.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)

    def get_trainer(self):
        return DenseTrainer(self)