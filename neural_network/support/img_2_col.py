from neural_network.gcpu import gcpu

def im2col(image, filter_size: tuple[int, int], stride: int = 1) -> gcpu.ndarray:
        batch, channels, height, width = image.shape
        fh, fw = filter_size

        output_height = int((height - fh) // stride + 1)
        output_width = int((width  - fw) // stride + 1)
        
        strides = image.strides
        stride_batch, stride_channel, stride_height, stride_width = strides

        cols = gcpu.lib.stride_tricks.as_strided(
            image,
            shape=(batch, channels, output_height, output_width, fh, fw),
            strides=(stride_batch, stride_channel, stride_height * stride, stride_width * stride, stride_height, stride_width)
        )

        cols = cols.reshape(batch * output_height * output_width, -1)
        return cols
