from neural_network.gcpu import gcpu
from neural_network.core.dense_network import DenseNetwork
from neural_network.foundation.kernel import Kernel
from neural_network.train import CnnTrainer
from neural_network.core.padding import Padding
from neural_network.storage import Storage
from typing import Union
from neural_network.support import im2col

class CnnNetwork(DenseNetwork):
    def __init__(self, config: dict, storage: Union[None, Storage]):
        self._mode = 'test'
        self._storage = storage
        self.cached_convolutions = []
        self.cached_pooling_indexes = []
        self.cached_bn = []

        assert config.get('processor') is not None, "Processor not defined"
        self.set_processor(config.get('processor'))
        
        self.padding_type: Padding = config.get('padding_type', Padding.SAME)
        self.input_shape = config.get('input_shape', (3, 50, 50))

        kernel_channel = self.input_shape[0]
        self._kernels: list[Kernel] = config.get('kernels', [])
        for kernel in self._kernels:
            kernel.initialize(kernel_channel)
            kernel_channel = kernel.number

        config['input_size'] = self._calculate_input_size(self.input_shape, self._kernels)
        super().__init__(config)

    def forward(self, x):
        self._reset_caches()
        convoluted = self._apply_convolutions(x)
        return super().forward(convoluted.reshape(x.shape[0], -1))

    def _reset_caches(self):
        self.cached_convolutions.clear()
        self.cached_pooling_indexes.clear()
        self.cached_bn.clear()

    def _apply_convolutions(self, x):
        output = x
        for kernel in self._kernels:
            output = self._apply_single_kernel(output, kernel)
        return output

    def _apply_single_kernel(self, input_layer, kernel: Kernel):
        conv_output = self._apply_single_convolution(input_layer, kernel)
        conv_output += kernel.bias()[:, gcpu.newaxis, gcpu.newaxis]

        if kernel.has_batch_normalization():
            conv_output = kernel.get_batch_normalization().batch_normalize(
                x=conv_output, mode=self._mode
            )

        if kernel.has_activation():
            conv_output = kernel.get_activation().activate(conv_output)

        if kernel.has_pooling():
            conv_output = kernel.get_pooling().apply_pooling(conv_output)

        self.cached_convolutions.append(conv_output)

        if self.is_trainning() and kernel.has_dropout():
            conv_output = self._apply_dropout(conv_output, kernel.get_dropout())

        return conv_output

    def _apply_single_convolution(self, input_layer, kernel: Kernel):
        _, _, i_h, i_w = input_layer.shape
        padding = self._get_padding((i_h, i_w), kernel.shape, kernel.stride)
        input_padded = self._add_padding(input_layer, padding)
        col = im2col(input_padded, kernel.shape, kernel.stride)

        conv_output = gcpu.einsum('ij,bj->bi', kernel.filters().reshape(kernel.number, -1), col)
        output_height, output_width = self._get_output_size(
            i_h, i_w, kernel.shape, kernel.stride, padding
        )

        return conv_output.reshape(input_layer.shape[0], kernel.number, output_height, output_width)

    def _calculate_input_size(self, input_shape: tuple[int, int, int], kernels: list[Kernel]) -> int:
        channels, height, width = input_shape
        for kernel in kernels:
            padding = self._get_padding((height, width), kernel.shape, kernel.stride)
            height, width = self._get_output_size(height, width, kernel.shape, kernel.stride, padding)

            if kernel.has_pooling():
                pooling = kernel.get_pooling()
                height, width = self._get_output_size(
                    height, width, pooling.shape, pooling.stride, ((0, 0), (0, 0))
                )

            channels = kernel.number

        return channels * height * width

    def _get_output_size(self, height: int, width: int, filter_shape: tuple[int, int], stride: int = 1, 
                         padding: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))) -> tuple[int, int]:
        pad_x = padding[0][0] + padding[0][1]
        pad_y = padding[1][0] + padding[1][1]
        output_height = gcpu.ceil(height + pad_x - filter_shape[0]) // stride + 1
        output_width = gcpu.ceil(width + pad_y - filter_shape[1]) // stride + 1
        return output_height, output_width

    def _add_padding(self, input_layer, padding: tuple[tuple[int, int], tuple[int, int]]) -> gcpu.ndarray:
        return gcpu.pad(
            input_layer,
            ((0, 0), (0, 0), padding[0], padding[1]),
            mode="constant",
            constant_values=0
        )

    def _get_padding(self, input_shape: tuple[int, int], filter_shape: tuple[int, int], stride: int = 1) -> tuple[int, int]:
        if self.padding_type == Padding.SAME:
            output_height = int(gcpu.ceil(input_shape[0] / stride))
            output_width = int(gcpu.ceil(input_shape[1] / stride))

            total_pad_x = max((output_height - 1) * stride + filter_shape[0] - input_shape[0], 0)
            total_pad_y = max((output_width - 1) * stride + filter_shape[1] - input_shape[1], 0)

            pad_x_top = total_pad_x // 2
            pad_x_bottom = total_pad_x - pad_x_top
            pad_y_left = total_pad_y // 2
            pad_y_right = total_pad_y - pad_y_left
            return (pad_x_top, pad_x_bottom), (pad_y_left, pad_y_right)

        return ((0, 0), (0, 0))

    def backward(self, x, y, output):
        dense_deltas = super().backward(
            self.cached_convolutions[-1].reshape(x.shape[0], -1), y, output
        )
        delta_conv = dense_deltas.reshape(self.cached_convolutions[-1].shape)

        filter_gradients = self._compute_filter_gradients(x, delta_conv)

        for i, kernel in enumerate(self._kernels):
            optimizer = kernel.get_optimizer() or self._global_optimizer
            kernel.update_filters(optimizer.update(f"kernel_{i}", kernel.filters(), filter_gradients[i]))

        return dense_deltas

    def _compute_filter_gradients(self, x, delta_conv):
        filter_gradients = []

        for i in range(len(self._kernels) - 1, -1, -1):
            kernel = self._kernels[i]
            input_layer = x if i == 0 else self.cached_convolutions[i - 1]
            filters = kernel.filters()
            num_filters, input_channels, fh, fw = filters.shape
            conv = self.cached_convolutions[i]
            optimizer = kernel.get_optimizer() or self._global_optimizer

            if kernel.has_activation():
                delta_conv *= kernel.get_activation().derivate(conv)

            grad_bias = gcpu.sum(delta_conv, axis=(0, 2, 3))
            kernel.update_bias(optimizer.update(f"kernel_bias_{i}", kernel.bias(), grad_bias))

            padding = self._get_padding(
                (input_layer.shape[2], input_layer.shape[3]), (fh, fw), kernel.stride
            )

            if kernel.has_pooling():
                delta_conv = kernel.get_pooling().unpooling(delta_conv)

            if kernel.has_batch_normalization():
                bn = kernel.get_batch_normalization()
                delta_conv, dgamma, dbeta = bn.batch_norm_backward(delta_conv)
                bn.update_gama(optimizer.update(f'bn_gamma_{i}', bn.get_gama(), dgamma))
                bn.update_beta(optimizer.update(f'bn_beta_{i}', bn.get_beta(), dbeta))

            batch_size, _, output_h, output_w = delta_conv.shape
            input_padded = self._add_padding(input_layer, padding)
            input_reshaped = im2col(input_padded, (fh, fw), kernel.stride)
            delta_reshaped = delta_conv.reshape(batch_size * output_h * output_w, num_filters)
            grad_filter = gcpu.dot(delta_reshaped.T, input_reshaped).reshape(filters.shape)
            filter_gradients.append(grad_filter)
            grad_filter += self.regularization_lambda * filters

            delta_col = delta_reshaped @ gcpu.flip(filters.reshape(num_filters, -1), axis=1)
            delta_conv = delta_col.reshape(batch_size, output_h, output_w, input_channels, fh, fw)
            delta_conv = delta_conv.transpose(0, 3, 4, 5, 1, 2).sum(axis=(2, 3))

        filter_gradients.reverse()
        return filter_gradients

    def train(self, x_batch, y_batch) -> gcpu.ndarray:
        output_batch = self.forward(x_batch)
        self.backward(x_batch, y_batch, output_batch)
        return output_batch

    def save_state(self):
        if self._storage:
            self._storage.store(self)

    def predict(self, x) -> gcpu.ndarray:
        return self.forward(x)

    def get_trainer(self):
        return CnnTrainer(self, self.get_processor())