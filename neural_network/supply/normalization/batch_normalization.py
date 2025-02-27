from neural_network.gcpu import driver

class BatchNormalization:
    def __init__(self, 
        num_filters: int, 
        axis: int = -1,
        gamma: float = 1.0,
        beta: float = 0.0,
        momentum: float = 0.99, 
        epsilon: float = 1e-5,
        center: bool = True, 
        scale: bool = True,
        trainable: bool = True
    ):
        self.axis = axis
        self.trainable = trainable
        self.center = center
        self.scale = scale
        self._epsilon = epsilon
        self.momentum = momentum
        
        param_shape = [1] * 4
        param_shape[axis] = num_filters
        param_shape = tuple(param_shape)

        self._gamma = driver.gcpu.full(param_shape, gamma) if scale else driver.gcpu.ones(param_shape)
        self._beta = driver.gcpu.full(param_shape, beta) if center else driver.gcpu.zeros(param_shape)
        
        self.running_mean = driver.gcpu.zeros(param_shape)
        self.running_var = driver.gcpu.ones(param_shape)
        
        self.cached_bn = None

    def forward(self, x, training=False):
        positive_axis = self.axis % x.ndim
        reduction_axes = tuple(i for i in range(x.ndim) if i != positive_axis)
        
        if training:
            batch_mean = driver.gcpu.mean(x, axis=reduction_axes, keepdims=True)
            batch_var = driver.gcpu.var(x, axis=reduction_axes, keepdims=True, ddof=0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        inv = driver.gcpu.reciprocal(driver.gcpu.sqrt(batch_var + self._epsilon))
        normalized = (x - batch_mean) * inv
        
        if self.scale:
            normalized *= self._gamma
        if self.center:
            normalized += self._beta

        self.cached_bn = (x, normalized, batch_mean, batch_var, inv, reduction_axes)
        return normalized

    def backward(self, dout):
        if self.cached_bn is None:
            raise RuntimeError("No cached batch normalization data for backward pass.")

        x, normalized, _, _, inv, reduction_axes = self.cached_bn
        m = driver.gcpu.prod(driver.gcpu.array([x.shape[i] for i in reduction_axes]))
        
        dgamma = driver.gcpu.sum(dout * normalized, axis=reduction_axes, keepdims=True) if self.scale and self.trainable else None
        dbeta = driver.gcpu.sum(dout, axis=reduction_axes, keepdims=True) if self.center and self.trainable else None
        
        dx_hat = dout * self._gamma if self.scale else dout
        dx = (1. / m) * inv * (m * dx_hat - driver.gcpu.sum(dx_hat, axis=reduction_axes, keepdims=True) - normalized * driver.gcpu.sum(dx_hat * normalized, axis=reduction_axes, keepdims=True))

        return dx, dgamma, dbeta
    
    def get_gamma(self):
        return self._gamma

    def update_gamma(self, gamma):
        self._gamma = gamma

    def get_beta(self):
        return self._beta

    def update_beta(self, beta):
        self._beta = beta
