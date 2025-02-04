from neural_network.gcpu import gcpu

class Dropout:
    def __init__(self, rate):
        assert 0 <= rate < 1, "Dropout rate must be in the range [0, 1)."
        self.rate: float = rate
        self._mask = None

    def apply(self, activations: gcpu.ndarray):
        retain_prob = 1 - self.rate
        self._mask = (gcpu.random.random(size=activations.shape) < retain_prob).astype(activations.dtype)
        
        return (activations * self._mask) / retain_prob

    def get_mask(self):
        return self._mask

    def scale_correction(self, gradients):
        if self._mask is None:
            raise ValueError("Dropout mask is None. Ensure `apply()` was called during forward pass.")
        
        if  self._mask.shape != gradients.shape:
            raise ValueError(f"Dropout mask shape {self._mask.shape} does not match gradients shape {gradients.shape}")
   
        return (gradients * self._mask) / (1 - self.rate)
