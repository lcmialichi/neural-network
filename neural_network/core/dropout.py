from neural_network.gcpu import driver

class Dropout:
    def __init__(self, rate):
        assert 0 <= rate < 1, "Dropout rate must be in the range [0, 1)."
        self.rate: float = rate
        self._mask = None

    def forward(self, activations):
        retain_prob = 1 - self.rate
        self._mask = (driver.gcpu.random.random(size=activations.shape) < retain_prob).astype(activations.dtype)
        return (activations * self._mask) / retain_prob

    def backwards(self, gradients):
        if self._mask is None:
            raise ValueError("Dropout mask is None. Ensure `apply()` was called during forward pass.")
        
        if  self._mask.shape != gradients.shape:
            raise ValueError(f"Dropout mask shape {self._mask.shape} does not match gradients shape {gradients.shape}")
   
        return  (gradients * self._mask)
