from neural_network.gcpu import driver
from neural_network.core.padding import Padding
from .image import im2col

def conv(input_layer, filters, number: int, stride: int, shape: tuple[int, int], padding_type: Padding):
    expected_filter_shape = (number, input_layer.shape[1], shape[0], shape[1])
    assert filters.shape == expected_filter_shape, f"Filters shape {filters.shape} != {expected_filter_shape}"
    
    batch_size, _, i_h, i_w = input_layer.shape  
    padding = get_padding((i_h, i_w), shape, stride, padding_type)
    input_padded = add_padding(input_layer, padding)
    col = im2col(input_padded, shape, stride, for_conv=True)
    filters_reshaped = filters.reshape(number, -1)
    conv_output = driver.gcpu.matmul(filters_reshaped, col.T)
    output_height, output_width = get_output_size(i_h, i_w, shape, stride, padding)
    conv_output = conv_output.T.reshape(batch_size, number, output_height, output_width)
    return conv_output

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
        ((0, 0), (0, 0), padding[0], padding[1]),
        mode="constant",
        constant_values=0
    )
    
def get_output_size(height: int, width: int, filter_shape: tuple[int, int], stride: int = 1, 
                         padding: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))) -> tuple[int, int]:
        pad_x = padding[0][0] + padding[0][1]
        pad_y = padding[1][0] + padding[1][1]
        output_height = (height + pad_x - filter_shape[0]) // stride + 1
        output_width = (width + pad_y - filter_shape[1]) // stride + 1
        return int(output_height), int(output_width)