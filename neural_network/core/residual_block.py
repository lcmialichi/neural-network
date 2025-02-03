from foundation.kernel import Kernel
from neural_network.initializations import He
from neural_network.activations import Relu

class ResidualBlock:
    def __init__(self, filters, kernel_size, stride, padding_type):
        self.conv1 = Kernel(number=filters, shape=kernel_size, stride=stride)
        self.conv1.initializer(He())
        self.conv1.activation(Relu())
        self.conv1.batch_normalization()

        self.conv2 = Kernel(number=filters, shape=kernel_size, stride=stride)
        self.conv2.initializer(He())
        self.conv2.activation(Relu())
        self.conv2.batch_normalization()

        self.padding_type = padding_type

    def forward(self, x):
        # Salva a entrada para a conexão residual
        self.residual = x

        # Aplica a primeira convolução
        out, _ = self.conv1.apply(x, self.padding_type)

        # Aplica a segunda convolução
        out, _ = self.conv2.apply(out, self.padding_type)

        # Soma a entrada residual à saída
        out += self.residual

        # Aplica ReLU após a soma
        out = Relu().activate(out)

        return out

    def backward(self, delta_conv):
        # Gradiente da ativação ReLU
        delta_conv *= Relu().derivate(self.residual + self.conv2.conv())

        # Gradiente da conexão residual
        delta_residual = delta_conv

        # Gradiente das camadas convolucionais
        delta_conv = self.conv2.backward(
            self.conv1.conv(), delta_conv, regularization_lambda, self.padding_type, mode
        )
        delta_conv = self.conv1.backward(
            self.residual, delta_conv, regularization_lambda, self.padding_type, mode
        )

        # Soma o gradiente da conexão residual ao gradiente das convoluções
        delta_conv += delta_residual

        return delta_conv