from neural_network.gcpu import driver
from neural_network.core.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.95, beta2=0.98, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param_name: str, param, grad):
        if param_name not in self.m:
            self.m[param_name] = driver.gcpu.zeros_like(param)
            self.v[param_name] = driver.gcpu.zeros_like(param)

        self.t += 1

        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad

        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
        
        param_update = -self.learning_rate * m_hat / (driver.gcpu.sqrt(v_hat) + self.epsilon)

        return param + param_update
