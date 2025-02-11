from neural_network.gcpu import driver
from neural_network.core.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(learning_rate)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = {}

    def update(self, param_name: str, param, grad, weight_decay: bool = True):
        if self.weight_decay != 0 and weight_decay:
            grad = grad + self.weight_decay * param

        if self.momentum != 0:
            if param_name not in self.velocity:
                self.velocity[param_name] = driver.gcpu.zeros_like(param)
            
            self.velocity[param_name] = (
                self.momentum * self.velocity[param_name] 
                + (1 - self.dampening) * grad
            )

            if self.nesterov:
                param_update = -self.learning_rate * (self.velocity[param_name] * self.momentum + grad)
            else:
                param_update = -self.learning_rate * self.velocity[param_name]
        else:
            param_update = -self.learning_rate * grad

        return param + param_update