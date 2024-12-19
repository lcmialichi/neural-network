import numpy as np
from neural_network.core.dense_network import DenseNetwork
from neural_network.initializations import Xavier
from neural_network.train import CnnTrainer
from neural_network.core.padding import Padding
from neural_network.configuration.cnn_configuration import CnnConfiguration
from neural_network.core import Initialization

class CnnNetwork(DenseNetwork):
    def __init__(
        self, 
        options: CnnConfiguration
    ):
        config = options.get_config()
        initializer: Initialization = config.get('initializer', Xavier())
        
        self.filters_options = config.get('filters', [])
        self.stride: int = config.get('stride', 1)
        self.padding_type: Padding = config.get('padding_type', Padding.SAME)
        self.input_shape = config.get('input_shape', (3, 50, 50))
        self.filters = initializer.generate_filters(self.filters_options, self.input_shape[0])
        config['input_size'] = self.get_input_size(self.input_shape, self.filters_options)
        self.cached_convolutions = []
        super().__init__(config, initializer=initializer)
        
        
    @staticmethod
    def config(config: dict = {})-> CnnConfiguration:
        return CnnConfiguration(config)
     
    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        self.cached_convolutions = []
        batch_normalized = self.batch_normalize(self.convolve_im2col(x, self.filters, self.stride, dropout))
        activated_output = self.activation.activate(batch_normalized)
        return super().forward(activated_output.reshape(x.shape[0], -1), dropout)

    def im2col(self, image, filter_size, stride):
        batch, channels, height, width = image.shape
        fh, fw = filter_size
        output_height = (height - fh) // stride + 1
        output_width = (width - fw) // stride + 1

        col = np.zeros((batch, channels, fh, fw, output_height, output_width))
     
        for y in range(fh):
            y_max = y + stride * output_height
            for x in range(fw):
                x_max = x + stride * output_width
                col[:, :, y, x, :, :] = image[:, :, y:y_max:stride, x:x_max:stride]

        return col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * output_height * output_width, channels * fh * fw)
    
    def convolve_im2col(self, input, filters_list: list, stride: int, apply_dropout: bool = False):
        output = input
        batch_size, channels, _, _ = output.shape
        for filters in filters_list:
            num_filters, input_channels, fh, fw = filters.shape

            assert channels == input_channels, (
                f"Erro: Canais de entrada ({channels}) não coincidem com os filtros ({input_channels})"
            )

            padded_input = self.add_padding(output,fh, fw)
    
            col = self.im2col(padded_input, (fh, fw), stride)
            filters_reshaped = filters.reshape(num_filters, -1).T
            conv_output = col @ filters_reshaped
            
            output_height = (padded_input.shape[2] - fh) // stride + 1
            output_width = (padded_input.shape[3] - fw) // stride + 1

            conv_output = conv_output.reshape(batch_size,num_filters, output_height, output_width)
        
            self.cached_convolutions.append(conv_output)
            if apply_dropout:
                conv_output = self.apply_dropout(conv_output)
                
            output = conv_output
            channels = num_filters 
            
        return output

    def get_input_size(self, input_shape: tuple[int, int, int], filters: list[dict]) -> int:
        channels, height, width = input_shape

        for filter_layer in filters:
            num_filters = filter_layer['number']
            filter_height, filter_width = filter_layer['shape']

            pad_x, pad_y = self.get_padding(self.padding_type, filter_height, filter_width)

            output_height = (height + 2 * pad_x - filter_height) // self.stride + 1
            output_width = (width + 2 * pad_y - filter_width) // self.stride + 1

            height, width = output_height, output_width
            channels = num_filters

        return channels * height * width

    def batch_normalize(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + 1e-8 
        return (x - mean) / std

    def add_padding(self, input: np.ndarray, filter_height: int, filter_width: int) -> np.ndarray:
        pad_x, pad_y = self.get_padding(self.padding_type, filter_height, filter_width)
        return np.pad(
            input,
            ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)),
            mode="constant",
            constant_values=0
        )
    
    def get_padding(self, padding: Padding, filter_height: int, filter_width: int) -> tuple[int, int]:
        _, input_height, input_width = self.input_shape

        if padding == Padding.SAME and self.stride > 1:
            pad_x = ((input_height - 1) * self.stride + filter_height - input_height) // 2
            pad_y = ((input_width - 1) * self.stride + filter_width - input_width) // 2
        elif padding == Padding.SAME and self.stride == 1 :
            pad_x = (filter_height - 1) // 2
            pad_y = (filter_width - 1) // 2
        else:
            pad_x, pad_y = 0, 0

        return pad_x, pad_y
    
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        filter_gradients = []

        flattened_last_conv = self.cached_convolutions[-1].reshape(x.shape[0], -1)
        dense_deltas = super().backward(flattened_last_conv, y, output)
        delta_conv = dense_deltas.reshape(self.cached_convolutions[-1].shape)

        for i in range(len(self.filters) - 1, -1, -1):
            input_layer = x if i == 0 else self.cached_convolutions[i - 1]
            num_filters, input_channels, fh, fw = self.filters[i].shape
            batch_size, _, output_h, output_w = delta_conv.shape

            delta_conv *= self.activation.derivate(self.cached_convolutions[i])

            input_padded = self.add_padding(input_layer, fh, fw)
            input_reshaped = self.im2col(input_padded, (fh, fw), self.stride)

            delta_reshaped = delta_conv.reshape(batch_size * output_h * output_w, num_filters)
            
            grad_filter = (delta_reshaped.T @ input_reshaped).reshape(self.filters[i].shape)
            filter_gradients.append(grad_filter)

            filters_reshaped = self.filters[i].reshape(num_filters, -1)
            rotated_filters = np.flip(filters_reshaped, axis=1)
            
            delta_col = delta_reshaped @ rotated_filters

            delta_conv = delta_col.reshape(batch_size, output_h, output_w, input_channels, fh, fw)
            delta_conv = delta_conv.transpose(0, 3, 4, 5, 1, 2).sum(axis=(2, 3))

        filter_gradients.reverse()

        for i in range(len(self.filters)):
            self.filters[i] = self.optimizer.update(f"filter_{i}", self.filters[i], filter_gradients[i])

        return dense_deltas

    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output_batch = self.forward(x_batch, True)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def get_trainer(self):
        return CnnTrainer(self)
    