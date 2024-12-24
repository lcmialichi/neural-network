import numpy as np
from neural_network.core.dense_network import DenseNetwork
from neural_network.initializations import Xavier
from neural_network.train import CnnTrainer
from neural_network.core.padding import Padding
from neural_network.core import Initialization
from neural_network.core import Activation
import matplotlib.pyplot as plt

class CnnNetwork(DenseNetwork):
    def __init__(self, config: dict):
        self.initializer: Initialization = config.get('initializer', Xavier())
        self.filters_options = config.get('filters', [])
        self.padding_type: Padding = config.get('padding_type', Padding.SAME)
        self.input_shape = config.get('input_shape', (3, 50, 50))
        self.filters = self.initializer.generate_filters(self.filters_options, self.input_shape[0])
        config['input_size'] = self._calculate_input_size(self.input_shape, self.filters_options)
        self.cached_convolutions = []
        self.cached_pooling_indexes = []
        super().__init__(config, initializer=self.initializer)
        
    def visualize_convolution_output(self, input, conv_output, title="Convolução"):
        example = conv_output[0]
        
        num_filters = example.shape[0]
        
        cols = 4  
        rows = (num_filters + cols - 1) // cols
        
        plt.ion()
        fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(12, 12)
        fig.suptitle(title, fontsize=16)
        
        axes = fig.subplots(rows + 1, cols)
        
        axes[0, 0].imshow(np.transpose(input[0], (1, 2, 0)), cmap='viridis')  
        axes[0, 0].set_title("Input")
        axes[0, 0].axis('off')
        
        for i in range(1, rows * cols + 1):
            ax = axes[i // cols, i % cols]
            
            if i - 1 < num_filters:
                ax.imshow(example[i - 1], cmap='viridis')
                ax.set_title(f"Filtro {i}")
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)


    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        self.cached_convolutions = []
        self.cached_pooling_indexes = []
        convoluted = self._apply_convolutions(x, dropout=dropout)
        return super().forward(convoluted.reshape(x.shape[0], -1), dropout)

    def _apply_convolutions(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        output = x
        for index, filters in enumerate(self.filters):
            options: dict = self.filters_options[index]
            output = self._apply_single_convolution(output, filters, options['stride'])
            activation: Activation = options['activation']
            output = activation.activate(output)
            if options.get('polling', False):
                output, indexes = self._apply_max_pooling(output, options['polling']['shape'], options['polling']['stride'])
                self.cached_pooling_indexes.append(indexes)
            self.cached_convolutions.append(output)
            if dropout:
                output = self._apply_dropout(output)
                
        # self.visualize_convolution_output(x, output)
        return output

    def _apply_single_convolution(self, input: np.ndarray, filters: np.ndarray, stride: int = 1) -> np.ndarray:
        _, _, i_h, i_w = input.shape
        padding = self._get_padding((i_h, i_w), (filters.shape[2], filters.shape[3]), stride)
        
        padded_input = self._add_padding(input, filters.shape[2:], stride)
        col = self._im2col(padded_input, filters.shape[2:], stride)
        
        filters_reshaped = filters.reshape(filters.shape[0], -1)
        
        conv_output = np.einsum('ij,bj->bi', filters_reshaped, col)
        
        conv_output = self._batch_normalize(conv_output)
        
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

        row_offsets = (max_idxs // pooling_shape[1]) + np.arange(0, output_height * stride, stride)[:, None]
        col_offsets = (max_idxs % pooling_shape[1]) + np.arange(0, output_width * stride, stride)[None, :]

        pooled_indexes = np.stack([row_offsets, col_offsets], axis=-1)

        return max_vals, pooled_indexes


    def _unpooling(self, grad, pool_cache, pool_shape: tuple[int, int], stride: int):
        batch_size, channels, pooled_height, pooled_width = grad.shape
        pool_height, pool_width = pool_shape

        original_height = pooled_height * stride + pool_height - stride
        original_width = pooled_width * stride + pool_width - stride

        unpooled_grad = np.zeros((batch_size, channels, original_height, original_width))

        row_indices = pool_cache[..., 0]
        col_indices = pool_cache[..., 1]

        batch_indices, channel_indices = np.meshgrid(
            np.arange(batch_size), np.arange(channels), indexing="ij"
        )
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

    def _batch_normalize(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + 1e-8
        return (x - mean) / std
    
    def _add_padding(self, input: np.ndarray, filter_shape: tuple[int, int], stride: int = 1) -> np.ndarray:
        pad_x, pad_y = self._get_padding((input.shape[2], input.shape[3]), filter_shape, stride)
        return np.pad(
            input,
            ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)),
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
            if delta_conv.shape != conv.shape:
                conv = self._resize_to_match(conv, delta_conv.shape)
            
            delta_conv *= activation.derivate(conv)
            if 'polling' in options:
                pooling_shape = options['polling']['shape']
                pooling_stride = options['polling']['stride']
                pool_cache = self.cached_pooling_indexes.pop()
                delta_conv = self._unpooling(delta_conv, pool_cache, pooling_shape, pooling_stride)
            
            batch_size, _, output_h, output_w = delta_conv.shape
            input_padded = self._add_padding(input_layer, (fh, fw), options.get('stride'))
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
    
    def _resize_to_match(self, data: np.ndarray, target_shape: tuple) -> np.ndarray:
        from skimage.transform import resize

        batch_size, channels, target_h, target_w = target_shape
        resized_data = np.zeros((batch_size, channels, target_h, target_w), dtype=data.dtype)

        for b in range(batch_size):
            for c in range(channels):
                resized_data[b, c] = resize(data[b, c], (target_h, target_w), mode='reflect', anti_aliasing=True)
        
        return resized_data

    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output_batch = self.forward(x_batch, dropout=True)
        self.backward(x_batch, y_batch, output_batch)
        
        if self.initializer.save_data():
            self.initializer.store(bias=self.biases, filters=self.filters, layers=self.weights)
            
        return output_batch
    
    def predict(self, x) -> np.ndarray:
        return self.softmax(self.forward(x))
    
    def get_trainer(self):
        return CnnTrainer(self)
