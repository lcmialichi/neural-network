from neural_network.gcpu import gcpu

class BatchNormalization:
    def __init__(self, num_filters: int,  gama: float = 1.0, beta: float = 0.0, momentum: float = 0.9):
        assert momentum > 0 and momentum < 1, "Momentum must be between 0 and 1"

        self.cached_bn = None
        self._gama = gcpu.ones((1, num_filters, 1, 1)) * gama
        self._beta = gcpu.zeros((1, num_filters, 1, 1)) + beta
        self.running_mean = gcpu.zeros((1, num_filters, 1, 1))
        self.running_var = gcpu.ones((1, num_filters, 1, 1))
        self.momentum = momentum

    def get_gama(self):
        return self._gama
    
    def update_gama(self, gama):
        self._gama = gama

    def get_beta(self):
        return self._beta

    def update_beta(self, beta):
        self._beta = beta

    def batch_normalize(self, x, mode: str = 'test') -> gcpu.ndarray:
        if mode in 'train':
            batch_mean = gcpu.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = gcpu.var(x, axis=(0, 2, 3), keepdims=True)

            x_hat = (x - batch_mean) / gcpu.sqrt(batch_var + 1e-8)
            out = self._gama * x_hat + self._beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.cached_bn = (x, x_hat, batch_mean, batch_var, self._gama)
        else:
            x_hat = (x - self.running_mean) / gcpu.sqrt(self.running_var + 1e-8)
            out = self._gama * x_hat + self._beta

        return out
    
    def batch_norm_backward(self, dout) -> gcpu.ndarray:
        x, x_hat, mean, var, gamma = self.cached_bn
        N, _, H, W = x.shape
        dgamma = gcpu.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
        dbeta = gcpu.sum(dout, axis=(0, 2, 3), keepdims=True)

        dx_hat = dout * gamma
        dvar = gcpu.sum(dx_hat * (x - mean) * -0.5 * gcpu.power(var + 1e-8, -1.5), axis=(0, 2, 3), keepdims=True)
        dmean = gcpu.sum(dx_hat * -1 / gcpu.sqrt(var + 1e-8), axis=(0, 2, 3), keepdims=True) + dvar * gcpu.sum(-2 * (x - mean), axis=(0, 2, 3), keepdims=True) / (N * H * W)
        dx = dx_hat / gcpu.sqrt(var + 1e-8) + dvar * 2 * (x - mean) / (N * H * W) + dmean / (N * H * W)
        
        return dx, dgamma, dbeta