import numpy as np
from PIL import Image
from typing import Tuple

class CnnNetwork:
    def __init__(
        self, 
        filter_size: tuple[int, int], 
        input_size: tuple[int, int, int], 
        stride: int = 1, 
        padding_type: str = "SAME", 
        num_filters: int = 3  # Adiciona o número de filtros
    ):
        fx, fy = filter_size
        _, ix, iy = input_size

        if fx > ix: fx = ix
        if fy > iy: fy = iy

        self.padding_type = padding_type
        self.stride: int = stride
        self.filter_size: tuple[int, int] = (fx, fy)
        self.input_size: tuple[int, int, int] = input_size  # (channels, height, width)
        self.num_filters = num_filters  # Número de filtros

        # Cria filtros com o formato (num_filters, fx, fy)
        self.filter = np.random.randn(num_filters, fx, fy)  # Aqui temos 3 filtros para RGB
        
    def forward(self, input: np.ndarray):
        padded_input = self.add_padding(input)  # Adiciona o padding na entrada
        convolved = self.convolve(padded_input)  # Aplica a convolução na entrada com padding
        
        return convolved

    def convolve(self, matrix: np.ndarray):
        fx, fy = self.filter_size
        channels, ix, iy = matrix.shape  # Pega os canais, altura e largura da entrada com padding

        # Calculando o tamanho da saída baseado no stride e padding
        output_height = (ix - fx) // self.stride + 1
        output_width = (iy - fy) // self.stride + 1
        output = np.zeros((self.num_filters, output_height, output_width))  # Tamanho da saída ajustado

        # Convolução com múltiplos filtros
        for f in range(self.num_filters):  # Iterar sobre os filtros
            for x in range(0, output_height):
                for y in range(0, output_width):
                    sum_result = 0
                    for c in range(channels):  # Iterar sobre os canais (RGB)
                        # Calcula os limites da região (evita ultrapassar os limites da matriz)
                        start_x = x * self.stride
                        start_y = y * self.stride
                        end_x = start_x + fx  # Limita a região para não ultrapassar a altura
                        end_y = start_y + fy  # Limita a região para não ultrapassar a largura

                        # Extrai a região considerando os limites calculados
                        region = matrix[c, start_x:end_x, start_y:end_y]

                        # Aplica o filtro na região extraída
                        sum_result += np.sum(region * self.filter[f, :region.shape[0], :region.shape[1]])

                    output[f, x, y] = sum_result  # Armazena o resultado final na posição (x, y)

        return output

    def add_padding(self, rgb_matrix: np.ndarray):
        fx, fy = self.filter_size
        _, ix, iy = self.input_size  # Tamanho da entrada

        if self.padding_type == "SAME":
            # Para stride = 1 ou maior, calculamos o padding de forma diferente
            if self.stride == 1:
                pad_x = (fx - 1) // 2  # Padding para altura com stride 1
                pad_y = (fy - 1) // 2  # Padding para largura com stride 1
            else:  # Para stride > 1
                pad_x = np.ceil(((ix - 1) * self.stride + fx - ix) / 2).astype(int)
                pad_y = np.ceil(((iy - 1) * self.stride + fy - iy) / 2).astype(int)
        else:  # Padding "VALID"
            pad_x = 0
            pad_y = 0

        # Armazena as variáveis de padding no objeto
        self.pad_x = pad_x
        self.pad_y = pad_y

        # Adiciona o padding nas bordas
        return np.pad(rgb_matrix, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode='constant', constant_values=0)
    
    def backpropagate(self, input: np.ndarray, output: np.ndarray, target: np.ndarray, learning_rate: float):
        """
        Realiza a retropropagação para ajustar os filtros.
        """
        # Função de perda MSE (Erro Quadrático Médio)
        error = output - target
        grad_output = error  # Derivada do MSE em relação à saída é o erro (dL/dy)

        # Gradiente de convolução
        grad_filter = np.zeros_like(self.filter)

        # Propagação do gradiente para os filtros
        for f in range(self.num_filters):
            for x in range(output.shape[1]):  # Itera pela altura
                for y in range(output.shape[2]):  # Itera pela largura
                    # Extrai a região correspondente da entrada para o filtro
                    start_x = x * self.stride
                    start_y = y * self.stride
                    end_x = start_x + self.filter_size[0]
                    end_y = start_y + self.filter_size[1]

                    region = input[:, start_x:end_x, start_y:end_y]

                    # Gradiente dos filtros
                    grad_filter[f] += grad_output[f, x, y] * region

        # Atualização dos filtros
        self.filter -= learning_rate * grad_filter

def process_image(image_path: str, filter_size: Tuple[int, int], stride: int = 1, num_filters: int = 3):
    # Carregar a imagem
    img = Image.open(image_path)
    
    # Convertê-la para RGB (caso seja uma imagem com 1 ou 4 canais)
    img = img.convert('RGB')
    
    # Converter a imagem para uma matriz numpy (canais, altura, largura)
    img_data = np.array(img)
    img_data = np.transpose(img_data, (2, 0, 1))  # De (altura, largura, canais) para (canais, altura, largura)

    channels, ix, iy = img_data.shape  # Captura os canais, altura e largura da imagem

    # Criação da rede CNN com a imagem
    cnn = CnnNetwork(filter_size, (channels, ix, iy), stride, "SAME", num_filters)

    # Passar a imagem pela CNN
    output = cnn.forward(img_data)
    # Função de perda: suponha que o valor alvo seja 1 (por exemplo, classificação binária ou regressão)
    target = np.ones_like(output)
    # Retropropagação
    cnn.backpropagate(img_data, np.array([0,1]), np.array([1,0]), learning_rate=0.01)

    # Normalizando o resultado de volta para a faixa de valores de pixel (0-255)
    output = np.clip(output, 0, 255).astype(np.uint8)
    print(output.shape)  # Verifica a forma da saída
    # Transformar a matriz de volta em uma imagem (usando o primeiro filtro para visualização)
    output_image = Image.fromarray(output[0])  # Apenas o primeiro filtro para visualização

    return output_image


# Testando com uma imagem no mesmo diretório
image_path = "a.png"  # Coloque o nome da sua imagem aqui

output_image = process_image(image_path, filter_size=(3, 3), stride=1, num_filters=1)

# Exibir a imagem resultante
output_image.show()

# Salvar a imagem resultante
output_image.save("output_image.jpg")
