from neural_network.core import Activation
from neural_network.gcpu import driver

class Gelu(Activation):
    def activate(self, x):
        return 0.5 * x * (1 + driver.gcpu.tanh(driver.gcpu.sqrt(2 / driver.gcpu.pi) * (x + 0.044715 * x ** 3)))

    def derivate(self, x):
        tanh_val = driver.gcpu.tanh(driver.gcpu.sqrt(2 / driver.gcpu.pi) * (x + 0.044715 * x ** 3))
        return 0.5 * (1 + tanh_val) + x * (1 - tanh_val ** 2) * driver.gcpu.sqrt(2 / driver.gcpu.pi) * (1 + 0.044715 * 3 * x ** 2)
