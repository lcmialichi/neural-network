from neural_network.gcpu import driver
from neural_network.core.padding import Padding
from .image import im2col

def conv(input_layer, filters, stride: int, padding_type: Padding):
    batch_size = input_layer.shape[0]
    in_height, in_width = input_layer.shape[1], input_layer.shape[2]
    fh, fw, _, out_channels = filters.shape
    padding = get_padding((in_height, in_width), (fh, fw), stride, padding_type)
    input_padded = add_padding(input_layer, padding)
    col = im2col(input_padded, (fh, fw), stride)
    filters_reshaped = filters.reshape(-1, out_channels)
    conv_output = driver.gcpu.matmul(col, filters_reshaped)
    output_height = (in_height + padding[0][0] + padding[0][1] - fh) // stride + 1
    output_width = (in_width + padding[1][0] + padding[1][1] - fw) // stride + 1
    
    return conv_output.reshape(batch_size, output_height, output_width, out_channels)

def get_padding(input_shape: tuple[int, int], filter_shape: tuple[int, int], stride: int = 1, padding_type = Padding.SAME) -> tuple[int, int]:
        if padding_type == Padding.SAME:
            output_height = int(driver.gcpu.ceil(input_shape[0] / stride))
            output_width = int(driver.gcpu.ceil(input_shape[1] / stride))

            total_pad_x = max((output_height - 1) * stride + filter_shape[0] - input_shape[0], 0)
            total_pad_y = max((output_width - 1) * stride + filter_shape[1] - input_shape[1], 0)

            pad_x_top = total_pad_x // 2
            pad_x_bottom = total_pad_x - pad_x_top
            pad_y_left = total_pad_y // 2
            pad_y_right = total_pad_y - pad_y_left
            return (pad_x_top, pad_x_bottom), (pad_y_left, pad_y_right)

        return ((0, 0), (0, 0))
    
def add_padding(input_layer, padding: tuple[tuple[int, int], tuple[int, int]]):
    return driver.gcpu.pad(
        input_layer,
        ((0, 0), padding[0], padding[1], (0, 0)),
        mode="constant",
        constant_values=0
    )

def get_output_size(input_size: tuple[int, int], filter_size: tuple[int, int], 
                   stride: int, padding: tuple[tuple[int, int]]):
    return (
        (input_size[0] + padding[0][0] + padding[0][1] - filter_size[0]) // stride + 1,
        (input_size[1] + padding[1][0] + padding[1][1] - filter_size[1]) // stride + 1
    )