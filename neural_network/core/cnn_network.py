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
        super().__init__(config, initializer=initializer)
        
        
    @staticmethod
    def config()-> CnnConfiguration:
        return CnnConfiguration()
     
    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        self.cached_convolutions = self.convolve_im2col(x, self.filters, self.stride)
        activated_output = self.activation.activate(self.cached_convolutions)
        return super().forward(activated_output.reshape(x.shape[0], -1), dropout)

    def im2col(self, image, filter_size, stride):
        batch, channels, height, width = image.shape
        f, fw = filter_size
        output_height = (height - f) // stride + 1
        output_width = (width - fw) // stride + 1

        col = np.zeros((batch, channels, f, fw, output_height, output_width))

        for y in range(f):
            y_max = y + stride * output_height
            for x in range(fw):
                x_max = x + stride * output_width
                col[:, :, y, x, :, :] = image[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * output_height * output_width, -1)
        return col

    def convolve_im2col(self, input, filters_list: list, stride: int):
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

            conv_output = conv_output.reshape(batch_size, output_height, output_width, num_filters)
            output = conv_output.transpose(0, 3, 1, 2) 
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


    def add_padding(self, input: np.ndarray, filter_height: int, filter_width: int) -> np.ndarray:
   
        if self.padding_type == Padding.SAME: # for now only works with stride = 1
            pad_x = (filter_height - 1) // 2
            pad_y = (filter_width - 1) // 2
        else:  # Padding VALID
            pad_x, pad_y = 0, 0

        return np.pad(
            input,
            ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)),
            mode="constant",
            constant_values=0
        )
    
    def get_padding(self, padding: Padding, filter_height: int, filter_width: int) -> tuple[int, int]:
     
        _, input_height, input_width = self.input_shape

        if padding == Padding.SAME:
            pad_x = ((input_height - 1) * self.stride + filter_height - input_height) // 2
            pad_y = ((input_width - 1) * self.stride + filter_width - input_width) // 2
        else:
            pad_x, pad_y = 0, 0

        return pad_x, pad_y
            
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        batch_size = x.shape[0]
        
        dense_grad = super().backward(self.cached_convolutions.reshape(batch_size, -1), y, output)
        dense_grad = dense_grad.reshape(batch_size, self.cached_convolutions.shape[1],
                                        self.cached_convolutions.shape[2],
                                        self.cached_convolutions.shape[3])

        derivative = self.activation.derivate(self.cached_convolutions)

        for layer_idx in range(len(self.filters) - 1, -1, -1):
            filters = self.filters[layer_idx]
            _, _, fh, fw = filters.shape

            pad_x, pad_y = self.get_padding(self.padding_type, fh, fw)

            output_height = (x.shape[2] + 2 * pad_x - fh) // self.stride + 1
            output_width = (x.shape[3] + 2 * pad_y - fw) // self.stride + 1

            col = self.im2col(x, (fh, fw), self.stride)
            grad_output = (dense_grad * derivative).transpose(0, 2, 3, 1).reshape(batch_size * output_height * output_width, -1)
            
            grad_filters = grad_output.T.dot(col).reshape(filters.shape)

            for f in range(filters.shape[0]):
                self.filters[layer_idx][f] = self.optimizer.update(f"filter_{layer_idx}_{f}", self.filters[layer_idx][f], grad_filters[f])



    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output_batch = self.forward(x_batch, True)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def get_trainer(self):
        return CnnTrainer(self)
    