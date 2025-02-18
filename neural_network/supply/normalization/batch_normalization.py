from neural_network.gcpu import driver

class BatchNormalization:
    def __init__(self, 
        num_filters: int, 
        axis: int = 1,
        gamma: float = 1.0,
        beta: float = 0.0, 
        momentum: float = 0.99, 
        epsilon: float = 0.001, 
        center: bool = True, 
        scale: bool = True,
        trainable: bool = True
    ):
        self.axis = axis
        self.trainable = trainable
        self.cached_bn = None
        self.center = center
        self.scale = scale

        param_shape = [1] * 4
        param_shape[self.axis] = num_filters
        param_shape = tuple(param_shape)

        self._gamma = driver.gcpu.ones(param_shape) * gamma if scale else driver.gcpu.ones(param_shape)
        self._beta = driver.gcpu.zeros(param_shape) + beta if center else driver.gcpu.zeros(param_shape)

        self._epsilon = epsilon
        self.running_mean = driver.gcpu.zeros(param_shape)
        self.running_var = driver.gcpu.ones(param_shape)
        self.momentum = momentum
        
    def get_gamma(self):
        return self._gamma
    
    def update_gamma(self, gamma):
        self._gamma = gamma

    def get_beta(self):
        return self._beta

    def update_beta(self, beta):
        self._beta = beta
        
    def batch_normalize(self, x, training=False):
        reduction_axes = tuple([i for i in range(x.ndim) if i != self.axis])
        m = driver.gcpu.prod([x.shape[i] for i in reduction_axes])
        
        if training and self.trainable:
            batch_mean = driver.gcpu.mean(x, axis=reduction_axes, keepdims=True)
            batch_var = driver.gcpu.var(x, axis=reduction_axes, keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        x_hat = (x - batch_mean) / driver.gcpu.sqrt(batch_var + self._epsilon)
        self.cached_bn = (x, x_hat, batch_mean, batch_var, reduction_axes, m)

        out = x_hat * self._gamma + self._beta
        return out
    
    def batch_norm_backward(self, dout):
        if self.cached_bn is None:
            raise RuntimeError("Backward chamado sem forward em modo de treino.")

        x, x_hat, batch_mean, batch_var, reduction_axes, m = self.cached_bn

        dgamma = driver.gcpu.sum(dout * x_hat, axis=reduction_axes, keepdims=True) if self.scale else driver.gcpu.zeros_like(self._gamma)
        dbeta = driver.gcpu.sum(dout, axis=reduction_axes, keepdims=True) if self.center else driver.gcpu.zeros_like(self._beta)
        
        dx_hat = dout * self._gamma
        inv_std = 1.0 / driver.gcpu.sqrt(batch_var + self._epsilon)
        
        dx = (1.0 / m) * inv_std * (m * dx_hat - driver.gcpu.sum(dx_hat, axis=reduction_axes, keepdims=True) - x_hat * driver.gcpu.sum(dx_hat * x_hat, axis=reduction_axes, keepdims=True))
        
        return dx, dgamma, dbeta
