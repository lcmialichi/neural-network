import numpy as np
from neural_network.core.dense_network import DenseNetwork
from neural_network.initializations import Xavier
from neural_network.train import CnnTrainer
from neural_network.core.padding import Padding
from neural_network.core import Initialization
from neural_network.core import Activation

class CnnNetwork(DenseNetwork):
    def __init__(self, config: dict):
        initializer: Initialization = config.get('initializer', Xavier())
        
        self.filters_options = config.get('filters', [])
        self.stride: int = config.get('stride', 1)
        self.padding_type: Padding = config.get('padding_type', Padding.SAME)
        self.input_shape = config.get('input_shape', (3, 50, 50))
        
        self.filters = initializer.generate_filters(self.filters_options, self.input_shape[0])
        config['input_size'] = self._calculate_input_size(self.input_shape, self.filters_options)
        
        self.cached_convolutions = []
        super().__init__(config, initializer=initializer)
    
    
    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        self.cached_convolutions = []
        convoluted = self._apply_convolutions(x, dropout=dropout)
        batch_normalized = self._batch_normalize(convoluted)
        return super().forward(batch_normalized.reshape(x.shape[0], -1), dropout)
    
    def _apply_convolutions(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        output = x
        for index, filters in enumerate(self.filters):
            output = self._apply_single_convolution(output, filters, dropout)
            activation: Activation = self.filters_options[index]['activation']
            
            output = activation.activate(output)
        return output
    
    def _apply_single_convolution(self, input: np.ndarray, filters: np.ndarray, dropout: bool) -> np.ndarray:
        padded_input = self._add_padding(input, filters.shape[2:])
        col = self._im2col(padded_input, filters.shape[2:])
        filters_reshaped = filters.reshape(filters.shape[0], -1).T
        conv_output = col @ filters_reshaped
        
        output_height = (padded_input.shape[2] - filters.shape[2]) // self.stride + 1
        output_width = (padded_input.shape[3] - filters.shape[3]) // self.stride + 1
        conv_output = conv_output.reshape(input.shape[0], filters.shape[0], output_height, output_width)
        
        self.cached_convolutions.append(conv_output)
        
        if dropout:
            conv_output = self._apply_dropout(conv_output)
        return conv_output

    def _calculate_input_size(self, input_shape: tuple[int, int, int], filters: list[dict]) -> int:
        channels, height, width = input_shape
        for filter_layer in filters:
            height, width = self._get_output_size(height, width, filter_layer['shape'])
            channels = filter_layer['number']
        return channels * height * width

    def _batch_normalize(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + 1e-8
        return (x - mean) / std
    
    def _add_padding(self, input: np.ndarray, filter_shape: tuple[int, int]) -> np.ndarray:
        pad_x, pad_y = self._get_padding(filter_shape)
        return np.pad(
            input,
            ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)),
            mode="constant",
            constant_values=0
        )
    
    def _get_padding(self, filter_shape: tuple[int, int]) -> tuple[int, int]:
        if self.padding_type == Padding.SAME:
            pad_x = (filter_shape[0] - 1) // 2
            pad_y = (filter_shape[1] - 1) // 2
        else:
            pad_x, pad_y = 0, 0
        return pad_x, pad_y
    
    def _im2col(self, image: np.ndarray, filter_size: tuple[int, int]) -> np.ndarray:
        batch, channels, height, width = image.shape
        fh, fw = filter_size
        output_height = (height - fh) // self.stride + 1
        output_width = (width - fw) // self.stride + 1
        
        col = np.zeros((batch, channels, fh, fw, output_height, output_width))
        for y in range(fh):
            y_max = y + self.stride * output_height
            for x in range(fw):
                x_max = x + self.stride * output_width
                col[:, :, y, x, :, :] = image[:, :, y:y_max:self.stride, x:x_max:self.stride]
        
        return col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * output_height * output_width, -1)
    
    def _get_output_size(self, height: int, width: int, filter_shape: tuple[int, int]) -> tuple[int, int]:
        pad_x, pad_y = self._get_padding(filter_shape)
        output_height = (height + 2 * pad_x - filter_shape[0]) // self.stride + 1
        output_width = (width + 2 * pad_y - filter_shape[1]) // self.stride + 1
        return output_height, output_width

    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        filter_gradients = []

        flattened_last_conv = self.cached_convolutions[-1].reshape(x.shape[0], -1)
        dense_deltas = super().backward(flattened_last_conv, y, output)
        delta_conv = dense_deltas.reshape(self.cached_convolutions[-1].shape)

        for i in range(len(self.filters) - 1, -1, -1):
            activation: Activation = self.filters_options[i]['activation']

            input_layer = x if i == 0 else self.cached_convolutions[i - 1]
            num_filters, input_channels, fh, fw = self.filters[i].shape
            batch_size, _, output_h, output_w = delta_conv.shape

            delta_conv *= activation.derivate(self.cached_convolutions[i])

            input_padded = self._add_padding(input_layer, (fh, fw))
            input_reshaped = self._im2col(input_padded, (fh, fw))

            delta_reshaped = delta_conv.reshape(batch_size * output_h * output_w, num_filters)
            
            grad_filter = (delta_reshaped.T @ input_reshaped).reshape(self.filters[i].shape)
            filter_gradients.append(grad_filter)

            delta_col = delta_reshaped @ np.flip(self.filters[i].reshape(num_filters, -1), axis=1)

            delta_conv = delta_col.reshape(batch_size, output_h, output_w, input_channels, fh, fw)
            delta_conv = delta_conv.transpose(0, 3, 4, 5, 1, 2).sum(axis=(2, 3))

        filter_gradients.reverse()

        for i in range(len(self.filters)):
            self.filters[i] = self.optimizer.update(f"filter_{i}", self.filters[i], filter_gradients[i])

        return dense_deltas
    
    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output_batch = self.forward(x_batch, dropout=True)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch
    
    def predict(self, x) -> np.ndarray:
        return self.forward(x)
    
    def get_trainer(self):
        return CnnTrainer(self)
