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
        batch_size, height, width, channels = input.shape
        self.input_shape = input.shape

        output_height = (height - self.shape[0]) // self.stride + 1
        output_width = (width - self.shape[1]) // self.stride + 1

        col = im2col(input, self.shape, self.stride)
        col_reshaped = col.reshape(batch_size, output_height, output_width, -1, channels)
        max_vals = driver.gcpu.max(col_reshaped,axis=3)
        max_idxs = driver.gcpu.argmax(col_reshaped, axis=3)

        self.cached_pooling_indexes = max_idxs

        return max_vals
        
    def unpooling(self, grad):
        if self.cached_pooling_indexes is None:
            raise RuntimeError("No cached max-pooling indexes available for unpooling.")
        
        batch_size, output_height, output_width, channels = grad.shape
        _, input_height, input_width, _ = self.input_shape

        unpooled_grad = driver.gcpu.zeros(self.input_shape)

        # Obter índices relativos (dentro da janela)
        row_indices_rel, col_indices_rel = driver.gcpu.unravel_index(
            self.cached_pooling_indexes, 
            (self.shape[0], self.shape[1])
        )

        # Calcular índices absolutos na entrada
        window_starts_i = (driver.gcpu.arange(output_height) * self.stride).reshape(1, -1, 1, 1)
        window_starts_j = (driver.gcpu.arange(output_width) * self.stride).reshape(1, 1, -1, 1)

        row_indices_abs = window_starts_i + row_indices_rel
        col_indices_abs = window_starts_j + col_indices_rel

        # Garantir que os índices não ultrapassem os limites
        row_indices_abs = driver.gcpu.clip(row_indices_abs, 0, input_height - 1)
        col_indices_abs = driver.gcpu.clip(col_indices_abs, 0, input_width - 1)

        # Distribuir os gradientes
        driver.gcpu.add.at(
            unpooled_grad,
            (
                driver.gcpu.arange(batch_size)[:, None, None, None],  # Índices do batch
                row_indices_abs,  # Índices das linhas
                col_indices_abs,  # Índices das colunas
                driver.gcpu.arange(channels)[None, None, None, :]  # Índices dos canais
            ),
            grad  # Gradiente (sem dimensão extra)
        )

        return unpooled_grad