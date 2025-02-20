from neural_network.gcpu import driver

class BatchNormalization:
  from neural_network.gcpu import driver

class BatchNormalization:
    def __init__(self, num_features: int, epsilon=1e-3, momentum=0.99, axis=-1):
        self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = driver.gcpu.ones((1, 1, 1, num_features))
        self.beta = driver.gcpu.zeros((1, 1, 1, num_features))
        
        self.running_mean = driver.gcpu.zeros((1, 1, 1, num_features))
        self.running_var = driver.gcpu.ones((1, 1, 1, num_features))
        
        self.cache = None


    def get_gamma(self):
        return self._gamma

    def update_gamma(self, gamma):
        self._gamma = gamma

    def get_beta(self):
        return self._beta

    def update_beta(self, beta):
        self._beta = beta

    def forward(self, x, training=True):
        if training:
            mean = driver.gcpu.mean(x, axis=(0, 1, 2), keepdims=True)
            var = driver.gcpu.var(x, axis=(0, 1, 2), keepdims=True, ddof=0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_hat = (x - mean) / driver.gcpu.sqrt(var + self.epsilon)
        out = self.gamma * x_hat + self.beta
        
        if training:
            self.cache = (x, x_hat, mean, var)
        
        return out

    def backward(self, dout):
        x, x_hat, mean, var = self.cache
        m = x.shape[0] * x.shape[1] * x.shape[2]
        
        dgamma = driver.gcpu.sum(dout * x_hat, axis=(0, 1, 2), keepdims=True)
        dbeta = driver.gcpu.sum(dout, axis=(0, 1, 2), keepdims=True)
        
        inv_std = 1 / driver.gcpu.sqrt(var + self.epsilon)
        dx_hat = dout * self.gamma
        dvar = driver.gcpu.sum(dx_hat * (x - mean) * -0.5 * inv_std**3, axis=(0, 1, 2), keepdims=True)
        dmean = driver.gcpu.sum(dx_hat * -inv_std, axis=(0, 1, 2), keepdims=True) + dvar * driver.gcpu.sum(-2 * (x - mean), axis=(0, 1, 2), keepdims=True) / m
        
        dx = (dx_hat * inv_std) + (dvar * 2 * (x - mean) / m) + (dmean / m)
        
        return dx, dgamma, dbeta
