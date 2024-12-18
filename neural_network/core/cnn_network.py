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
    def config()-> CnnConfiguration:
        return CnnConfiguration()
     
    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:
        self.cached_convolutions = []
        activated_output = self.activation.activate(self.convolve_im2col(x, self.filters, self.stride))
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
            print(padded_input.shape, col.shape)
            conv_output = col @ filters_reshaped

            output_height = (padded_input.shape[2] - fh) // stride + 1
            output_width = (padded_input.shape[3] - fw) // stride + 1

            conv_output = conv_output.reshape(batch_size, output_height, output_width, num_filters)
            self.cached_convolutions.append(conv_output)
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
        batch_size, channels, height, width = x.shape
        
        dense_error = super().backward(
            self.cached_convolutions[-1].reshape(batch_size, -1), y, output
        )
        
        deltas = [dense_error.reshape(self.cached_convolutions[-1].shape).transpose(0, 3, 1, 2)]
        
        filters_grad = [np.zeros_like(filters) for filters in self.filters]
        
        for i in range(len(self.filters) - 1, -1, -1):
            filters = self.filters[i]
            num_filters, input_channels, fh, fw = filters.shape  
            current_conv = self.cached_convolutions[i].transpose(0, 3, 1, 2)
            
            padded_error = self.add_padding(deltas[-1], fh, fw)
            padded_input = self.add_padding(current_conv, fh, fw)
            print(padded_error.shape)
            exit()
            col_input = self.im2col(padded_input, (fh, fw), self.stride)
            col_error = self.im2col(padded_error, (fh, fw), self.stride)
            filters_reshaped = filters.reshape(num_filters, -1)
            print(col_error.shape, filters_reshaped.shape)
            exit()
            filters_grad[i] = col_input.T @ col_error
            
            deltas.append(np.dot(col_error, filters_reshaped.T))
        
        return deltas[0].dot(self.filters[0].T).reshape(x.shape)
            


    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output_batch = self.forward(x_batch, True)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def get_trainer(self):
        return CnnTrainer(self)
    