from neural_network.gcpu import gcpu

class Dropout:
    def __init__(self, rate):
        assert 0 <= rate < 1, "Dropout rate must be in the range [0, 1)."
        self.rate: float = rate
        self._mask = None

    def apply(self, activations: gcpu.ndarray):
        retain_prob = 1 - self.rate
        self._mask = (gcpu.random.random(size=activations.shape) < retain_prob).astype(activations.dtype)
        
        activations *= self._mask
        return activations / retain_prob

    def get_mask(self):
        return self._mask

    def scale_correction(self, gradients):
        gradients /= (1 - self.rate)
        gradients *= self._mask
        return gradients
