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
        filter_size: tuple[int, int] = (3, 3), 
        stride: int = 1, 
        padding_type: str = "SAME", 
        num_filters: int = 3
    ):
        super().__init__(config, initializer)
  
        self.filter_size = filter_size
        self.stride = stride
        self.padding_type = padding_type
        self.num_filters = num_filters
        
        self.filters = initializer.generate_filters(
            channels_number=config.get('input_channels', 3),
            filter_size=filter_size,
            num_filters=num_filters
        )
     

    def forward(self, x: np.ndarray, dropout: bool = False) -> np.ndarray:

        batch_size, _, _, _ = x.shape
       
        convolved_outputs = []
        for i in range(batch_size):
            convolved = self.convolve(x[i])
            convolved_outputs.append(convolved)
        
        self.cached_convolutions = np.array(convolved_outputs)
   
        return super().forward(self.cached_convolutions.reshape(batch_size, -1), dropout)

    def convolve(self, input: np.ndarray) -> np.ndarray:
        matrix = input
        if self.padding_type == "SAME":
            matrix = self.add_padding(matrix)

        fx, fy = self.filter_size
        _, ix, iy = matrix.shape

        output_height = (ix - fx) // self.stride + 1
        output_width = (iy - fy) // self.stride + 1

        output = np.zeros((self.num_filters, output_height, output_width))
     
        for f in range(self.num_filters):
            for x in range(output_height):
                for y in range(output_width):
                    start_x = x * self.stride
                    start_y = y * self.stride
                    end_x = start_x + fx
                    end_y = start_y + fy

                    if end_x > matrix.shape[1] or end_y > matrix.shape[2]:
                        continue

                    output[f, x, y] =  np.sum(matrix[:, start_x:end_x, start_y:end_y] * self.filters[f, :, :, :])
        return output


    def add_padding(self, rgb_matrix: np.ndarray):
        fx, fy = self.filter_size
        _, ix, iy = rgb_matrix 

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

        return np.pad(rgb_matrix, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode='constant', constant_values=0)
    
    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        batch_size = x.shape[0]

        if self.padding_type == "SAME":
            x_padded = np.array([self.add_padding(x[i]) for i in range(batch_size)])
        else:
            x_padded = x

        dense_grad = super().backward(self.cached_convolutions.reshape(batch_size, -1), y, output)

        grad_filters = np.zeros_like(self.filters)

        dense_grad = dense_grad.reshape(batch_size, self.num_filters,
                                        self.cached_convolutions.shape[2],
                                        self.cached_convolutions.shape[3])
    
        _, _, size_x, size_y = dense_grad.shape

        fx, fy = self.filter_size
        _, _ = x_padded.shape[2], x_padded.shape[3]

        for sx in range(size_x):
            for sy in range(size_y):
                start_x = sx * self.stride
                start_y = sy * self.stride
                end_x = start_x + fx
                end_y = start_y + fy

                region = x_padded[:, :, start_x:end_x, start_y:end_y]
                
                for f in range(self.num_filters):
                    grad_filters[f] += np.sum(dense_grad[:, f, sx, sy][:, np.newaxis, np.newaxis, np.newaxis] * region, axis=0)

        self.filters -= self.learning_rate * grad_filters


    def train(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        self.y_true.append(y_batch)
        output_batch = self.forward(x_batch, True)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def get_trainer(self):
        return CnnTrainer(self)