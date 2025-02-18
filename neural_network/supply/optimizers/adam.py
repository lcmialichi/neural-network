from neural_network.gcpu import driver
from neural_network.core.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 1

    def update(self, param_name: str, param, grad, weight_decay: bool = True):
        if param_name not in self.m:
            self.m[param_name] = driver.gcpu.zeros_like(param)
            self.v[param_name] = driver.gcpu.zeros_like(param)

        if self.weight_decay != 0 and weight_decay:
            grad = grad + self.weight_decay * param
            
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad

        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

        param_update = -self.learning_rate * m_hat / (driver.gcpu.sqrt(v_hat) + self.epsilon)

        return param + param_update
    
    def step(self):
        self.t += 1