from neural_network.foundation import Kernel
from neural_network.supply import He, Relu
from neural_network.foundation.block import Block

class ResidualBlock(Block):
    def __init__(self, number, shape, stride, downsample=False):
        super().__init__()
        
        self.conv1 = Kernel(number=number, shape=shape, stride=stride)
        self.conv1.initializer(He())
        self.conv1.activation(Relu())
        self.conv1.batch_normalization()

        self.conv2 = Kernel(number=number, shape=shape, stride=1)
        self.conv2.initializer(He())
        self.conv2.activation(Relu())
        self.conv2.batch_normalization()

        self.downsample = downsample
        if self.downsample:
            self.shortcut = Kernel(number=number, shape=(1, 1), stride=stride)
            self.shortcut.initializer(He())
            self.shortcut.batch_normalization()

        self.input = None
        self.out1 = None
        self.out2 = None

    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, x):
        self.input = x
        self.conv1.clone_hyper_params(self)
        self.conv2.clone_hyper_params(self)

        self.out1 = self.conv1.forward(x)
        self.out2 = self.conv2.forward(self.out1)
        
        shortcut = x
        if self.downsample:
            self.shortcut.clone_hyper_params(self)
            shortcut = self.shortcut.forward(x)

        out = self.out2 + shortcut
        self.store_logits(out)
        return Relu().activate(out)

    def backward(self, input_layer, y, delta_conv):
        delta_conv *= Relu().derivate(self.logits())
       
        delta_residual = delta_conv

        delta_conv = self.conv2.backward(self.out1, y, delta_conv)
        delta_conv = self.conv1.backward(self.input, y, delta_conv)

        if self.downsample:
            delta_shortcut = self.shortcut.backward(input_layer, y, delta_residual)
            delta_conv += delta_shortcut
        
        return delta_conv


