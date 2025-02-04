from neural_network.core.pooling import Pooling
from neural_network.gcpu import gcpu
from neural_network.support import im2col, col2im

class AvgPooling(Pooling):
    def __init__(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self.shape = shape
        self.stride = stride
        self.input_shape = None

    def apply_pooling(self, input):
        batch_size, channels, height, width = input.shape
        
        self.input_shape = (height, width)
        output_height = (height - self.shape[0]) // self.stride + 1
        output_width = (width - self.shape[1]) // self.stride + 1

        col = im2col(input, self.shape, self.stride)
        col = col.reshape(batch_size, channels, output_height, output_width, -1)

        avg_vals = gcpu.mean(col, axis=-1)

        return avg_vals

    def unpooling(self, grad):
        batch_size, channels, out_h, out_w = grad.shape
        pool_h, pool_w = self.shape

        grad_expanded = grad[:, :, :, :, gcpu.newaxis].repeat(pool_h * pool_w, axis=-1)
        grad_expanded = grad_expanded.reshape(batch_size, channels, -1)

        grad_expanded /= (pool_h * pool_w)

        unpooled_grad = col2im(
            grad_expanded,
            output_shape=self.input_shape,
            filter_shape=self.shape,
            stride=self.stride
        )

        return unpooled_grad