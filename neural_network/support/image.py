from neural_network.gcpu import driver

def im2col(image, filter_size: tuple[int, int], stride: int = 1):
        batch, channels, height, width = image.shape
        fh, fw = filter_size
        output_height = int((height - fh) // stride + 1)
        output_width = int((width  - fw) // stride + 1)
    
        strides = image.strides
        stride_batch, stride_channel, stride_height, stride_width = strides

        cols = driver.gcpu.lib.stride_tricks.as_strided(
            image,
            shape=(batch, channels, output_height, output_width, fh, fw),
            strides=(stride_batch, stride_channel, stride_height * stride, stride_width * stride, stride_height, stride_width)
        )

        cols = cols.reshape(batch * output_height * output_width, -1)
        return cols

def col2im(cols, output_shape, filter_size: tuple[int, int], stride: int):
    batch, channels, height, width = output_shape
    fh, fw = filter_size
    
    output_h = (height - fh) // stride + 1
    output_w = (width - fw) // stride + 1
    
    cols_reshaped = cols.reshape(batch, output_h, output_w, channels, fh, fw).transpose(0, 3, 1, 2, 4, 5)
    
    h_idx = stride * driver.gcpu.arange(output_h)[:, None, None, None]
    w_idx = stride * driver.gcpu.arange(output_w)[None, :, None, None]
    
    fh_idx = driver.gcpu.arange(fh)[None, None, :, None]
    fw_idx = driver.gcpu.arange(fw)[None, None, None, :]
    
    final_h = h_idx + fh_idx
    final_w = w_idx + fw_idx
    
    image = driver.gcpu.zeros((batch, channels, height, width))
    driver.gcpu.add.at(
        image,
        (
            slice(None),
            slice(None),
            final_h.ravel(),
            final_w.ravel()
        ),
        cols_reshaped.reshape(batch, channels, -1, fh * fw)
    )
    
    return image