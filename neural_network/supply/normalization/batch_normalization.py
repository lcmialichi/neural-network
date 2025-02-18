from neural_network.gcpu import driver

class BatchNormalization:
    def __init__(self, 
            num_filters: int, 
            gamma: float = 1.0,
            beta: float = 0.0, 
            momentum: float = 0.99, 
            epsilon: float = 0.001, 
            center: bool = True, 
            scale: bool = True
        ):
        assert 0 < momentum < 1, "Momentum deve estar entre 0 e 1"

        self.cached_bn = None
        self.center = center
        self.scale = scale

        self._gamma = driver.gcpu.ones((1, num_filters, 1, 1)) * gamma if scale else None
        self._beta = driver.gcpu.zeros((1, num_filters, 1, 1)) + beta if center else None

        self._epsilon = epsilon
        self.running_mean = driver.gcpu.zeros((1, num_filters, 1, 1))
        self.running_var = driver.gcpu.ones((1, num_filters, 1, 1))
        self.momentum = momentum

    def get_gamma(self):
        return self._gamma
    
    def update_gamma(self, gamma):
        self._gamma = gamma

    def get_beta(self):
        return self._beta

    def update_beta(self, beta):
        self._beta = beta

    def batch_normalize(self, x, mode: str = 'test'):
        if mode == 'train':
            batch_mean = driver.gcpu.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = driver.gcpu.var(x, axis=(0, 2, 3), keepdims=True)

            x_hat = (x - batch_mean) / driver.gcpu.sqrt(batch_var + self._epsilon)
            out = x_hat

            if self.scale:
                out = out * self._gamma
            if self.center:
                out = out + self._beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.cached_bn = (x, x_hat, batch_mean, batch_var)
        else:
            x_hat = (x - self.running_mean) / driver.gcpu.sqrt(self.running_var + self._epsilon)
            out = x_hat
            if self.scale:
                out = out * self._gamma
            if self.center:
                out = out + self._beta

        return out

    def batch_norm_backward(self, dout):
        assert self.cached_bn is not None, "Error: no cached value to use batch_norm_backward"
        
        x, x_hat, mean, var = self.cached_bn
        N, _, H, W = x.shape
        m = N * H * W

        dgamma = driver.gcpu.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True) if self.scale else None
        dbeta = driver.gcpu.sum(dout, axis=(0, 2, 3), keepdims=True) if self.center else None

        dx_hat = dout * self._gamma if self.scale else dout

        sqrt_var = driver.gcpu.sqrt(var + self._epsilon)
        dx = (1.0 / (m * sqrt_var)) * (m * dx_hat - driver.gcpu.sum(dx_hat, axis=(0,2,3), keepdims=True) - x_hat * driver.gcpu.sum(dx_hat * x_hat, axis=(0,2,3), keepdims=True))
        
        return dx, dgamma, dbeta
