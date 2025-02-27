from neural_network.gcpu import driver

def im2col(image, filter_size: tuple[int, int], stride: int):
    batch, height, width, channels = image.shape
    fh, fw = filter_size

    if fh > height or fw > width:
        raise ValueError("The filter size must be smaller than the image dimensions.")
    
    output_height = (height - fh) // stride + 1
    output_width = (width - fw) // stride + 1

    image_contig = driver.gcpu.ascontiguousarray(image)
    
    new_shape = (batch, output_height, output_width, fh, fw, channels)
    new_strides = (
        image_contig.strides[0],
        image_contig.strides[1] * stride,
        image_contig.strides[2] * stride,
        image_contig.strides[1],
        image_contig.strides[2],
        image_contig.strides[3]
    )
    
    cols_view = driver.gcpu.lib.stride_tricks.as_strided(
        image_contig, 
        shape=new_shape, 
        strides=new_strides,
    )
    
    return cols_view.copy()

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
        cols_reshaped.reshape(batch, channels, height, width)
    )
    
    return image