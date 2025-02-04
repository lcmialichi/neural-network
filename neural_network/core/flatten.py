class Flatten:
    def __init__(self):
        self.shape = None
        
    def forward(self, input):
        self.shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backward(self, input_layer, y, delta):
        return delta.reshape(self.shape)