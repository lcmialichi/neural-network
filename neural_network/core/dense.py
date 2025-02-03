from neural_network.gcpu import gcpu
from typing import Union
from neural_network.core.base_network import BaseNetwork
from neural_network.train import DenseTrainer
from neural_network.optimizers import Adam
from neural_network.foundation import HiddenLayer
from neural_network.foundation import Output

class Dense(BaseNetwork):

    hidden_outputs: list = []
    regularization_lambda: None
    loss_function : None
    dlogits: list = []
    _block_output: list = []

    def __init__(self):
        self._optimizer = None
        self._blocks: list = []
        self.mode = None

    def add_layer(
        self, 
        size: int, 
        dropout: Union[float, None] = None,
        ) -> "HiddenLayer":
        layer = HiddenLayer(size=size, dropout=dropout)
        self._blocks.append(layer)
        return layer
    
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, x: gcpu.ndarray) -> gcpu.ndarray:
        self._block_output.clear()
        output = x
        for block in self._blocks:
            if hasattr(block, 'mode'):
                block.mode = self.mode
            if hasattr(block, 'regularization_lambda'):
                block.regularization_lambda = self.regularization_lambda
            if hasattr(block, 'optimizer'):
                block.optimizer(self._optimizer)
                
            output = block.forward(output)
            self._block_output.append(output)

        return output

    def backward(self, x: gcpu.ndarray, y: gcpu.ndarray, output: gcpu.ndarray):
        if not self._loss_function:
            raise RuntimeError("Output loss function not defined")
        
      
        deltas_pred = self._loss_function.gradient(output, y)
        delta = deltas_pred
        for i in range(len(self._blocks) - 1, -1, -1):
            block = self._blocks[i]
            delta = block.backward(delta, self._block_output[i -1])
            print(delta.shape)
            exit()  
        
        return delta.reshape(x.shape)

    def _compute_deltas(self, y: gcpu.ndarray, output: gcpu.ndarray) -> list:
        deltas = []
        output_delta = self.loss_function.gradient(output, y)
        
        if self._output.do_derivative():
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