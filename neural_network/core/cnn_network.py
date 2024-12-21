import numpy as np
from neural_network.core.dense_network import DenseNetwork
from neural_network.initializations import Xavier
from neural_network.train import CnnTrainer
from neural_network.core.padding import Padding
from neural_network.core import Initialization
from neural_network.core import Activation

class CnnNetwork(DenseNetwork):
    def __init__(self, config: dict):
        self.initializer: Initialization = config.get('initializer', Xavier())
        self.filters_options = config.get('filters', [])
        self.stride: int = config.get('stride', 1)
        self.padding_type: Padding = config.get('padding_type', Padding.SAME)
        self.input_shape = config.get('input_shape', (3, 50, 50))
        self.filters = self.initializer.generate_filters(self.filters_options, self.input_shape[0])
        config['input_size'] = self._calculate_input_size(self.input_shape, self.filters_options)
        self.cached_convolutions = []
        self.cached_pooling_indexes = []
        super().__init__(config, initializer=self.initializer)

    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        self.cached_convolutions = []
        self.cached_pooling_indexes = []
        convoluted = self._apply_convolutions(x, dropout=dropout)
        return super().forward(convoluted.reshape(x.shape[0], -1), dropout)

    def _apply_convolutions(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        output = x
        for index, filters in enumerate(self.filters):
            options: dict = self.filters_options[index]
            output = self._apply_single_convolution(output, filters)
            activation: Activation = options['activation']
            output = activation.activate(output)
            if options.get('polling', False):
                output, indexes = self._apply_max_pooling(output, options['polling']['shape'], options['polling']['stride'])
                self.cached_pooling_indexes.append(indexes)
            self.cached_convolutions.append(output)
            if dropout:
                output = self._apply_dropout(output)
        return output

    def _apply_single_convolution(self, input: np.ndarray, filters: np.ndarray) -> np.ndarray:
        padded_input = self._add_padding(input, filters.shape[2:])
        col = self._im2col(padded_input, filters.shape[2:])
        filters_reshaped = filters.reshape(filters.shape[0], -1)
        
        conv_output = np.einsum('ij,bj->bi', filters_reshaped, col)
        conv_output = self._batch_normalize(conv_output)

        output_height = (padded_input.shape[2] - filters.shape[2]) // self.stride + 1
        output_width = (padded_input.shape[3] - filters.shape[3]) // self.stride + 1
        return conv_output.reshape(input.shape[0], filters.shape[0], output_height, output_width)

    def _apply_max_pooling(self, input: np.ndarray, pooling_shape: tuple[int, int] = (2, 2), stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        batch_size, channels, height, width = input.shape
        pool_height, pool_width = pooling_shape
        
        output_height = (height - pool_height) // stride + 1
        output_width = (width - pool_width) // stride + 1
        
        pooled_output = np.zeros((batch_size, channels, output_height, output_width))
        pooled_indexes = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=int)
        
        for i in range(output_height):
            for j in range(output_width):
                h_start, h_end = i * stride, i * stride + pool_height
                w_start, w_end = j * stride, j * stride + pool_width
                
                window = input[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(window, axis=(2, 3))
                max_idxs = np.argmax(window.reshape(batch_size, channels, -1), axis=2)
                
                pooled_output[:, :, i, j] = max_vals
                max_coords_h, max_coords_w = np.divmod(max_idxs, pool_width)
                pooled_indexes[:, :, i, j, 0] = h_start + max_coords_h
                pooled_indexes[:, :, i, j, 1] = w_start + max_coords_w
        
        return pooled_output, pooled_indexes

    def _unpooling(self, grad, pool_cache, pool_shape: tuple[int, int], stride: int):
        batch_size, channels, pooled_height, pooled_width = grad.shape
        pool_height, pool_width = pool_shape

        original_height = pooled_height * stride + pool_height - stride
        original_width = pooled_width * stride + pool_width - stride

        unpooled_grad = np.zeros((batch_size, channels, original_height, original_width))
        pooled_indices = pool_cache.reshape(batch_size, channels, -1, 2)

        for b in range(batch_size):
            for c in range(channels):
                grad_values = grad[b, c].reshape(-1)
                indices = pooled_indices[b, c].reshape(-1, 2)
                unpooled_grad[b, c, indices[:, 0], indices[:, 1]] = grad_values

        return unpooled_grad
    
    def _calculate_input_size(self, input_shape: tuple[int, int, int], filters: list[dict]) -> int:
        channels, height, width = input_shape
        for filter_layer in filters:
            height, width = self._get_output_size(height, width, filter_layer['shape'])

            if 'polling' in filter_layer:
                pooling_shape = filter_layer['polling']['shape']
                stride = filter_layer['polling']['stride']
                height, width = self._get_output_size(height=height, width=width, filter_shape=pooling_shape, stride=stride)
            
            channels = filter_layer['number']
        
        return channels * height * width
    
    def _get_output_size(self, height: int, width: int, filter_shape: tuple[int, int], stride: int = 1) -> tuple[int, int]:
        pad_x, pad_y = self._get_padding(filter_shape)
        output_height = (height + 2 * pad_x - filter_shape[0]) // stride + 1
        output_width = (width + 2 * pad_y - filter_shape[1]) // stride + 1
        return output_height, output_width

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
        stride = self.stride

        output_height = (height - fh) // stride + 1
        output_width = (width - fw) // stride + 1

        strides = image.strides
        stride_batch, stride_channel, stride_height, stride_width = strides

        cols = np.lib.stride_tricks.as_strided(
            image,
            shape=(batch, channels, output_height, output_width, fh, fw),
            strides=(stride_batch, stride_channel, stride_height * stride, stride_width * stride, stride_height, stride_width)
        )

        cols = cols.reshape(batch * output_height * output_width, -1)
        return cols

    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        filter_gradients = []

        dense_deltas = super().backward(self.cached_convolutions[-1].reshape(x.shape[0], -1), y, output)
        delta_conv = dense_deltas.reshape(self.cached_convolutions[-1].shape)

        for i in range(len(self.filters) - 1, -1, -1):
            options: dict = self.filters_options[i]
            activation: Activation = options.get('activation')

            input_layer = x if i == 0 else self.cached_convolutions[i - 1]
            num_filters, input_channels, fh, fw = self.filters[i].shape

            delta_conv *= activation.derivate(self.cached_convolutions[i])

            if 'polling' in options:
                pooling_shape = options['polling']['shape']
                pooling_stride = options['polling']['stride']
                pool_cache = self.cached_pooling_indexes.pop()
                delta_conv = self._unpooling(delta_conv, pool_cache, pooling_shape, pooling_stride)
            
            batch_size, _, output_h, output_w = delta_conv.shape
            input_padded = self._add_padding(input_layer, (fh, fw))
            input_reshaped = self._im2col(input_padded, (fh, fw))

            delta_reshaped = delta_conv.reshape(batch_size * output_h * output_w, num_filters)
            grad_filter = np.dot(delta_reshaped.T, input_reshaped).reshape(self.filters[i].shape)
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
        
        if self.initializer.save_data():
            self.initializer.store(
                bias=self.biases,
                filters=self.filters,
                layers=self.weights
            )
        return output_batch
    
    def predict(self, x) -> np.ndarray:
        return self.softmax(self.forward(x))
    
    def get_trainer(self):
        return CnnTrainer(self)
