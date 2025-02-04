from neural_network.gcpu import gcpu
from typing import Union
from neural_network.core.base_network import BaseNetwork
from neural_network.train import DenseTrainer
from neural_network.foundation import Layer

class Dense(BaseNetwork):

    hidden_outputs: list = []
    dlogits: list = []
    _block_output: list = []
    global_optimizer = None

    def __init__(self):
        self._optimizer = None
        self._blocks: list = []
        self.mode = None
        self.regularization_lambda = None
        self.loss_function = None

    def add_layer(
        self, 
        size: int, 
        dropout: Union[float, None] = None,
        ) -> "Layer":
        layer = Layer(size=size, dropout=dropout)
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
            if hasattr(block, 'global_optimizer'):
                block.global_optimizer = self.global_optimizer
                
            output = block.forward(output)
            self._block_output.append(output)

        return output

    def backward(self, input_layer: gcpu.ndarray, y, output: gcpu.ndarray):
        delta = self.loss_function.gradient(output, y)
        for i in range(len(self._blocks) - 1, -1, -1):
            block = self._blocks[i]
            prev_output = self._block_output[i - 1] if i > 0 else input_layer
            delta = block.backward(prev_output, y, delta)
        return delta.reshape(input_layer.shape)

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