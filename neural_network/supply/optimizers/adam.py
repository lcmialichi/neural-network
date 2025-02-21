from neural_network.gcpu import driver
from neural_network.core.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-07,
        weight_decay=0.0,
        amsgrad=False,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.m = {}
        self.v = {}
        self.v_hat = {} if amsgrad else None
        self.iterations = 0

    def update(self, param_name: str, param, grad, weight_decay: bool = True):
        if param_name not in self.m:
            self.m[param_name] = driver.gcpu.zeros_like(param)
            self.v[param_name] = driver.gcpu.zeros_like(param)
            if self.amsgrad:
                self.v_hat[param_name] = driver.gcpu.full_like(param, -driver.gcpu.inf)

        if self.weight_decay != 0 and weight_decay:
            param = param - self.learning_rate * self.weight_decay * param

        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

        t = self.iterations + 1
        m_hat = self.m[param_name] / (1 - driver.gcpu.power(self.beta1, t))
        v_hat = self.v[param_name] / (1 - driver.gcpu.power(self.beta2, t))

        if self.amsgrad:
            self.v_hat[param_name] = driver.gcpu.maximum(self.v_hat[param_name], v_hat)
            v_final = self.v_hat[param_name]
        else:
            v_final = v_hat

        param_update = -self.learning_rate * m_hat / (driver.gcpu.sqrt(v_final) + self.epsilon)
        return param + param_update

    def step(self):
        self.iterations += 1