import numpy as np
from neural_network.core.dense_network import DenseNetwork
from neural_network.initializations import Xavier
from neural_network.train import CnnTrainer
from neural_network.core.padding import Padding
from neural_network.core import Initialization
from neural_network.core import Activation

class CnnNetwork(DenseNetwork):
    
    cached_convolutions: list = []
    cached_pooling_indexes: list = []
    cached_bn: list = []
    
    def __init__(self, config: dict):
        self._mode = 'test'

        assert config.get('processor') is not None, "processor not defined"

        self.set_processor(config.get('processor'))
        self.initializer: Initialization = config.get('initializer', Xavier())
        self.filters_options = self.initializer.get_filters_options(config.get('filters'))
        self.padding_type: Padding = config.get('padding_type', Padding.SAME)
        self.input_shape = config.get('input_shape', (3, 50, 50))

        self.filters = self.initializer.generate_filters(self.filters_options, self.input_shape[0])
        self.kernel_biases = self.initializer.generate_kernel_bias(self.filters)

        config['input_size'] = self._calculate_input_size(self.input_shape, self.filters_options)
        
        super().__init__(config, initializer=self.initializer)
         
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cached_convolutions = []
        self.cached_pooling_indexes = []
        self.cached_bn = []
        convoluted = self._apply_convolutions(x)
        return super().forward(convoluted.reshape(x.shape[0], -1))

    def _apply_convolutions(self, x: np.ndarray) -> np.ndarray:
        output = x
        for index, filters in enumerate(self.filters):
            options: dict = self.filters_options[index]
            output = self._apply_single_convolution(output, filters, options['stride'])
            output += self.kernel_biases[index][:, np.newaxis, np.newaxis]
                                                
            if "bn" in options:
                output = self._batch_normalize(output, options['bn'])
                
            activation: Activation = options['activation']
            output = activation.activate(output)
            if 'polling' in options:
                output, indexes = self._apply_max_pooling(output, options['polling']['shape'], options['polling']['stride'])
                self.cached_pooling_indexes.append(indexes)
            
            
            self.cached_convolutions.append(output)
            if self._mode in 'train':
                output = self._apply_dropout(output)
                
        return output

    def _apply_single_convolution(self, input: np.ndarray, filters: np.ndarray, stride: int = 1) -> np.ndarray:
        _, _, i_h, i_w = input.shape
        padding = self._get_padding((i_h, i_w), (filters.shape[2], filters.shape[3]), stride)
        
        padded_input = self._add_padding(input, padding)
        col = self._im2col(padded_input, filters.shape[2:], stride)
        
        filters_reshaped = filters.reshape(filters.shape[0], -1)
        
        conv_output = np.einsum('ij,bj->bi', filters_reshaped, col)

        output_height, output_width = self._get_output_size(
            i_h, i_w, (filters.shape[2], filters.shape[3]), stride, padding
        )

        return conv_output.reshape(input.shape[0], filters.shape[0], output_height, output_width)


    def _apply_max_pooling(self, input: np.ndarray, pooling_shape: tuple[int, int] = (2, 2), stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        batch_size, channels, height, width = input.shape
        
        output_height, output_width = self._get_output_size(
            height, width, pooling_shape, stride, (0, 0)
        )

        col = self._im2col(input, pooling_shape, stride)
        col = col.reshape(batch_size, channels, output_height, output_width, -1)

        max_vals = np.max(col, axis=-1)
        max_idxs = np.argmax(col, axis=-1)

        row_indices = max_idxs // pooling_shape[1]
        col_indices = max_idxs % pooling_shape[1]

        row_offsets = np.arange(0, output_height * stride, stride)[:, None]
        col_offsets = np.arange(0, output_width * stride, stride)[None, :]

        row_indices += row_offsets
        col_indices += col_offsets

        pooled_indexes = np.stack([row_indices, col_indices], axis=-1)
        return max_vals, pooled_indexes

    def _unpooling(self, grad, pool_cache, size: tuple[int, int]):
        batch_size, channels, _, _ = grad.shape
        unpooled_grad = np.zeros((batch_size, channels, size[0], size[1]))
        
        batch_indices, channel_indices = np.meshgrid(
            np.arange(batch_size), np.arange(channels), indexing="ij"
        )
        
        row_indices = np.clip(pool_cache[..., 0], 0, size[0] - 1)
        col_indices = np.clip(pool_cache[..., 1], 0, size[1] - 1)

        batch_indices = batch_indices[..., None, None]
        channel_indices = channel_indices[..., None, None]

        unpooled_grad[batch_indices, channel_indices, row_indices, col_indices] = grad

        return unpooled_grad

    
    def _calculate_input_size(self, input_shape: tuple[int, int, int], filters: list[dict]) -> int:
        channels, height, width = input_shape
        for filter_layer in filters:
            height, width = self._get_output_size(
                height=height, 
                width=width, 
                filter_shape=filter_layer['shape'], 
                stride=filter_layer['stride'], 
                padding=self._get_padding(
                    (height, width),
                    filter_layer.get('shape'),
                    filter_layer.get('stride')
                )
            )

            if 'polling' in filter_layer:
                pooling_shape = filter_layer['polling']['shape']
                stride = filter_layer['polling']['stride']
                height, width = self._get_output_size(
                    height=height, 
                    width=width, 
                    filter_shape=pooling_shape, 
                    stride=stride, 
                    padding=(0, 0)
                )
            
            channels = filter_layer['number']
            
        return channels * height * width
    
    def _get_output_size(
        self,
        height: int, 
        width: int, 
        filter_shape: tuple[int, int], 
        stride: int = 1,
        padding: tuple[int, int] = (0, 0)
    ) -> tuple[int, int]:
        pad_x, pad_y = padding
        output_height = np.ceil((height + 2 * pad_x - filter_shape[0]) / stride) + 1
        output_width = np.ceil((width + 2 * pad_y - filter_shape[1]) / stride) + 1

        return int(output_height), int(output_width)

    def _batch_normalize(self, x: np.ndarray, bn_param: dict) -> np.ndarray:
        gamma, beta, momentum = bn_param['gamma'], bn_param['beta'], bn_param['momentum']
        running_mean, running_var = bn_param['running_mean'], bn_param['running_var']

        if self._mode in 'train':
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            x_hat = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
            out = gamma * x_hat + beta

            bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * batch_mean
            bn_param['running_var'] = momentum * running_var + (1 - momentum) * batch_var

            self.cached_bn.append((x, x_hat, batch_mean, batch_var, gamma))
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + 1e-8)
            out = gamma * x_hat + beta

        return out

    def _batch_norm_backward(self, dout: np.ndarray, cache: tuple) -> np.ndarray:
        x, x_hat, mean, var, gamma = cache
        N, _, H, W = x.shape

        dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        dx_hat = dout * gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * np.power(var + 1e-8, -1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(var + 1e-8), axis=(0, 2, 3), keepdims=True) + dvar * np.sum(-2 * (x - mean), axis=(0, 2, 3), keepdims=True) / (N * H * W)
        dx = dx_hat / np.sqrt(var + 1e-8) + dvar * 2 * (x - mean) / (N * H * W) + dmean / (N * H * W)

        return dx, dgamma, dbeta
    
    def _add_padding(self, 
            input: np.ndarray, 
            padding: tuple[int, int] = (0, 0)
        ) -> np.ndarray:
        return np.pad(
            input,
            ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
            mode="constant",
            constant_values=0
        )
    
    def _get_padding(self, input_shape: tuple[int, int], filter_shape: tuple[int, int], stride: int = 1) -> tuple[int, int]:
        if self.padding_type == Padding.SAME:
            output_height = np.ceil(input_shape[0] / stride)
            output_width = np.ceil(input_shape[1] / stride)
            total_pad_x = max((output_height - 1) * stride + filter_shape[0] - input_shape[0], 0)
            total_pad_y = max((output_width - 1) * stride + filter_shape[1] - input_shape[1], 0)

            pad_x = int(total_pad_x / 2)
            pad_y = int(total_pad_y / 2)
            
            return pad_x, pad_y
            
        return 0, 0
    
    def _im2col(self, image: np.ndarray, filter_size: tuple[int, int], stride: int = 1) -> np.ndarray:
        batch, channels, height, width = image.shape
        fh, fw = filter_size

        output_height = int(np.ceil((height  - fh) / stride) + 1)
        output_width = int(np.ceil((width  - fw) / stride) + 1)
        
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
            conv = self.cached_convolutions[i]
            
            grad_bias = np.sum(delta_conv, axis=(0, 2, 3))
            self.kernel_biases[i] = self.optimizer.update(f"kernel_bias_{i}", self.kernel_biases[i], grad_bias)

            delta_conv *= activation.derivate(conv)
            padding = self._get_padding((input_layer.shape[2], input_layer.shape[3]), (fh, fw), options.get('stride'))
            
            if 'polling' in options:
                pool_cache = self.cached_pooling_indexes.pop()
                delta_conv = self._unpooling(delta_conv, pool_cache, input_layer.shape[2:])
            
            if 'bn' in options:
                delta_conv, dgamma, dbeta = self._batch_norm_backward(delta_conv, self.cached_bn.pop())
                options['bn']['gamma'] = self.optimizer.update(
                    f"gamma_{i}", options['bn']['gamma'], dgamma
                )
                options['bn']['beta'] = self.optimizer.update(
                    f"beta_{i}", options['bn']['beta'], dbeta
                )
                
            batch_size, _, output_h, output_w = delta_conv.shape
            
            input_padded = self._add_padding(input_layer, padding)
            input_reshaped = self._im2col(input_padded, (fh, fw), options.get('stride'))

            delta_reshaped = delta_conv.reshape(batch_size * output_h * output_w, num_filters)
            grad_filter = np.dot(delta_reshaped.T, input_reshaped).reshape(self.filters[i].shape)
            filter_gradients.append(grad_filter)
            grad_filter += self.regularization_lambda * self.filters[i]

            delta_col = delta_reshaped @ np.flip(self.filters[i].reshape(num_filters, -1), axis=1)
            delta_conv = delta_col.reshape(batch_size, output_h, output_w, input_channels, fh, fw)
            delta_conv = delta_conv.transpose(0, 3, 4, 5, 1, 2).sum(axis=(2, 3))
          
        filter_gradients.reverse()
        for i in range(len(self.filters)):
            self.filters[i] = self.optimizer.update(f"filter_{i}", self.filters[i], filter_gradients[i])

        return dense_deltas
    
    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output_batch = self.forward(x_batch)
        self.backward(x_batch, y_batch, output_batch)
        
        if self.initializer.save_data():
            self.initializer.store(
                bias=self.biases, 
                filters=self.filters, 
                layers=self.weights, 
                filters_options=self.filters_options
            )
            
        return output_batch
    
    def predict(self, x) -> np.ndarray:
        return self.forward(x)
    
    def get_trainer(self):
        return CnnTrainer(self, self.get_processor())
