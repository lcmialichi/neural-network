from neural_network.core.pooling import Pooling
from neural_network.gcpu import driver
from neural_network.support import im2col

class MaxPooling(Pooling):
    def __init__(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self.shape = shape
        self.stride = stride
        self.cached_pooling_indexes = None
        self.input_shape = None

    def apply_pooling(self, input):
        batch_size, channels, height, width = input.shape
        
        self.input_shape = (height, width)
        output_height = (height  - self.shape[0]) // self.stride + 1
        output_width = (width  - self.shape[1]) // self.stride + 1

        col = im2col(input, self.shape, self.stride)
        col = col.reshape(batch_size, channels, output_height, output_width, -1)

        max_vals = driver.gcpu.max(col, axis=-1)
        max_idxs = driver.gcpu.argmax(col, axis=-1)

        row_indices = max_idxs // self.shape[1]
        col_indices = max_idxs % self.shape[1]

        row_offsets = driver.gcpu.arange(0, output_height * self.stride, self.stride)[:, None]
        col_offsets = driver.gcpu.arange(0, output_width * self.stride, self.stride)[None, :]

        row_indices += row_offsets
        col_indices += col_offsets
        
        row_indices = driver.gcpu.clip(row_indices + row_offsets, 0, height - 1)
        col_indices = driver.gcpu.clip(col_indices + col_offsets, 0, width - 1)

        pooled_indexes = driver.gcpu.stack([row_indices, col_indices], axis=-1)
        self.cached_pooling_indexes = pooled_indexes

        return max_vals
    
    def unpooling(self, grad):
        pool_cache = self.cached_pooling_indexes
        batch_size, channels, _, _ = grad.shape

        unpooled_grad = driver.gcpu.zeros((batch_size, channels, self.input_shape[0], self.input_shape[1]))
        
        batch_indices, channel_indices = driver.gcpu.meshgrid(
            driver.gcpu.arange(batch_size), driver.gcpu.arange(channels), indexing="ij"
        )
        
        row_indices = driver.gcpu.clip(pool_cache[..., 0], 0, self.input_shape[0] - 1)
        col_indices = driver.gcpu.clip(pool_cache[..., 1], 0, self.input_shape[1] - 1)

        batch_indices = batch_indices[..., None, None]
        channel_indices = channel_indices[..., None, None]

        unpooled_grad[batch_indices, channel_indices, row_indices, col_indices] = grad

        return unpooled_grad
