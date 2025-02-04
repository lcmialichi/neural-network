from neural_network.foundation import Kernel
from neural_network.supply import He, Relu, Softmax
from .residual_block import ResidualBlock

class ResNet:
    def __init__(self, num_classes=2, layers=[2, 2, 2, 2]): 
        
        self.initial = Kernel(number=64, shape=(7, 7), stride=2)
        self.initial.initializer(He())
        self.initial.activation(Relu())
        self.initial.batch_normalization()
        self.initial.max_pooling(shape=(3, 3), stride=2)

        # Definindo os estágios de blocos residuais
        self.stage1 = self._make_layer(64, (3, 3), layers[0], stride=1)
        self.stage2 = self._make_layer(128, (3, 3), layers[1], stride=2)
        self.stage3 = self._make_layer(256, (3, 3), layers[2], stride=2)
        self.stage4 = self._make_layer(512, (3, 3), layers[3], stride=2)

        # Camada de flatten e totalmente conectada
        self.flatten = Kernel(number=512, shape=(1, 1), stride=1)
        self.fc = Kernel(number=num_classes, shape=(1, 1), stride=1)
        self.fc.initializer(He())

        self.num_classes = num_classes

    def _make_layer(self, filters, shape, num_blocks, stride):
        layers = []
    
        layers.append(ResidualBlock(filters, shape, stride, downsample=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(filters, shape, stride=1))
        return layers

    def forward(self, x):
        x = self.initial.forward(x)

        for layer in self.stage1:
            x = layer.forward(x)

        for layer in self.stage2:
            x = layer.forward(x)

        for layer in self.stage3:
            x = layer.forward(x)

        for layer in self.stage4:
            x = layer.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)

        return Softmax().activate(x)

    def backward(self, x, y, delta_fc):
        delta_fc = self.fc.backward(x, y, delta_fc)

        delta_fc = self.flatten.backward(x, y, delta_fc)

        for stage in reversed([self.stage4, self.stage3, self.stage2, self.stage1]):
            for block in reversed(stage):
                delta_fc = block.backward(x, y, delta_fc)

        return delta_fc
