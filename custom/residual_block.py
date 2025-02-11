from neural_network.blocks import Kernel
from neural_network.supply import He, Relu
from neural_network.blocks.block import Block
from neural_network.core.pooling import Pooling
from neural_network.supply import pooling

class ResidualBlock(Block):
    _pooling = None
    def __init__(self, number, shape, stride, downsample=False):
        super().__init__()
        
        self.downsample = downsample
        self.conv1 = Kernel(number=number, shape=shape, stride=stride)
        self.conv1.initializer(He())
        self.conv1.activation(Relu())
        self.conv1.batch_normalization()

        self.conv2 = Kernel(number=number, shape=shape, stride=1)
        self.conv2.initializer(He())
        self.conv2.activation(Relu())
        self.conv2.batch_normalization()

        self.shortcut = None
       
        self.shortcut = Kernel(number=number, shape=(1, 1), stride=stride)
        self.shortcut.initializer(He())
        self.shortcut.batch_normalization()

        self.input = None
        self.out1 = None
        self.out2 = None

    def boot(self, shape: tuple):
        return

    def forward(self, x):
        self.input = x
        self.conv1.clone_hyper_params(self)
        self.conv2.clone_hyper_params(self)

        self.out1 = self.conv1.forward(x)
        self.out2 = self.conv2.forward(self.out1)

        shortcut = x
        if self.downsample or shortcut.shape != self.out2.shape:
            self.shortcut.clone_hyper_params(self)
            shortcut = self.shortcut.forward(x)

        out = self.out2 + shortcut
        self.store_logits(out)
        out = Relu().activate(out)
        
        if self.has_pooling():
            out = self.get_pooling().apply_pooling(out)

        return out

    def backward(self, input_layer, y, delta_conv):
        if self.has_pooling():
            delta_conv = self.get_pooling().unpooling(delta_conv)

        delta_conv *= Relu().derivate(self.logits())
        delta_shortcut = delta_conv
        if self.downsample:
            delta_shortcut = self.shortcut.backward(input_layer, y, delta_conv)

        delta_conv = self.conv2.backward(self.out1, y, delta_conv)
        delta_conv = self.conv1.backward(self.input, y, delta_conv)
        
        return delta_conv + delta_shortcut
    
    def has_pooling(self) -> bool:
        return self._pooling is not None
    
    def get_pooling(self) -> "Pooling":
        return self._pooling

    def max_pooling(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self._pooling = pooling.MaxPooling(shape=shape, stride=stride)
    
    def avg_pooling(self, shape: tuple[int, int] = (2, 2), stride: int = 1):
        self._pooling = pooling.AvgPooling(shape=shape, stride=stride)
