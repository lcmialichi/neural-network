import numpy as np
from neural_network.activations import Sigmoid
from neural_network.core import Activation
from neural_network.core.dense_network import DenseNetwork
from neural_network.initializations import Xavier
from neural_network.train import CnnTrainer

class CnnNetwork(DenseNetwork):
    def __init__(
        self, 
        config: dict, 
        initializer: Xavier = Xavier(), 
    ):
        super().__init__(config, initializer)
  
        self.filter_size = config.get('filter_size', (3, 3))
        self.stride = config.get('stride', 1)
        self.padding_type = config.get('padding_type', 'SAME')
        self.num_filters = config.get('num_filters', 3)
        
        self.filters = initializer.generate_filters(
            channels_number=config.get('input_channels', 3),
            filter_size=self.filter_size,
            num_filters=self.num_filters
        )
     

    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:

        self.cached_convolutions = np.array(self.convolve(x))
        return super().forward(self.cached_convolutions.reshape(x.shape[0], -1), dropout)

    def convolve(self, input: np.ndarray) -> np.ndarray:
        matrix = self.add_padding(input)

        fx, fy = self.filter_size
        b, _, ix, iy = matrix.shape

        output_height = (ix - fx) // self.stride + 1
        output_width = (iy - fy) // self.stride + 1

        output = np.zeros((b, self.num_filters, output_height, output_width))
     
        for f in range(self.num_filters):
            for x in range(output_height):
                for y in range(output_width):
                    start_x = x * self.stride
                    start_y = y * self.stride
                    end_x = start_x + fx
                    end_y = start_y + fy

                    output[:, f, x, y] =  np.sum(matrix[:,:, start_x:end_x, start_y:end_y] * self.filters[f])
        return output


    def add_padding(self, rgb_matrix: np.ndarray):
        fx, fy = self.filter_size
        _ ,_, ix, iy = rgb_matrix.shape

        if self.padding_type == "SAME":
            if self.stride == 1:
                pad_x = (fx - 1) // 2 
                pad_y = (fy - 1) // 2
            else:
                pad_x = np.ceil(((ix - 1) * self.stride + fx - ix) / 2).astype(int)
                pad_y = np.ceil(((iy - 1) * self.stride + fy - iy) / 2).astype(int)
        else:
            pad_x = 0
            pad_y = 0

        self.pad_x = pad_x
        self.pad_y = pad_y

        return np.pad(rgb_matrix, ((0,0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode='constant', constant_values=0)
    
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray):
        batch_size = x.shape[0]
        x_padded = self.add_padding(x)

        grad_filters = np.zeros_like(self.filters)
        dense_grad = super().backward(self.cached_convolutions.reshape(batch_size, -1), y, output)

        dense_grad = dense_grad.reshape(batch_size, self.num_filters,
                                        self.cached_convolutions.shape[2],
                                        self.cached_convolutions.shape[3])
    
        _, _, size_x, size_y = dense_grad.shape

        fx, fy = self.filter_size

        for sx in range(size_x):
            for sy in range(size_y):
                start_x = sx * self.stride
                start_y = sy * self.stride
                end_x = start_x + fx
                end_y = start_y + fy

                region = x_padded[:, :, start_x:end_x, start_y:end_y]
                    
                for f in range(self.num_filters):
                    grad_filters[f] += np.sum(
                        dense_grad[:, f, sx, sy][:, np.newaxis, np.newaxis, np.newaxis] * region,
                        axis=0
                    )

        self.filters -= self.learning_rate * grad_filters
        return grad_filters


    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        self.y_true.append(y_batch)
        output_batch = self.forward(x_batch, True)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def get_trainer(self):
        return CnnTrainer(self)