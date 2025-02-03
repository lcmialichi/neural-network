from . import Kernel
from scipy.ndimage import zoom

class SkipConnection():
    def __init__(self, current: Kernel, skip: Kernel):
        self._current = current
        self._skip = skip
        self._skip_grad = None

    def apply(self, x):
        skip_output = self._skip.conv()
        
        if skip_output.shape[2:] != x.shape[2:]:
            skip_output = self.interpolate(skip_output, x.shape[2:])

        return skip_output + x  
    
    def skip_kernel(self) -> Kernel:
        return self._skip

    def interpolate(self, image, target_shape):
        scale_factors = (target_shape[0] / image.shape[2], target_shape[1] / image.shape[3])
        return zoom(image, (1, 1, *scale_factors), order=1) 
