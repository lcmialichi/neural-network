import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    """Calcula a aproximação numérica do gradiente usando diferenças finitas."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        x[idx] = old_value + eps
        f_plus = f(x)
        
        x[idx] = old_value - eps
        f_minus = f(x)
        
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x[idx] = old_value
        
        it.iternext()
    
    return grad

# Teste para gradientes da convolução
def test_convolution_gradients(conv_layer, input_tensor, target_output):
    """Testa os gradientes da camada convolucional comparando com diferenças finitas."""
    def loss_fn(filters):
        conv_layer.update_filters(filters)
        output = conv_layer.forward(input_tensor)
        return np.sum((output - target_output) ** 2)
    
    computed_grad = conv_layer.backward(input_tensor, target_output, np.ones_like(target_output))
    numerical_grad = numerical_gradient(loss_fn, conv_layer.filters())
    
    error = np.linalg.norm(computed_grad - numerical_grad) / (np.linalg.norm(computed_grad) + 1e-7)
    print(f"Erro relativo nos gradientes da convolução: {error}")
    assert error < 1e-3, "Os gradientes da convolução parecem incorretos!"
    print("Gradientes da convolução passaram no teste.")

# Teste para batch normalization
def test_batch_norm_gradients(bn_layer, input_tensor):
    """Testa os gradientes do batch normalization."""
    def loss_fn_gamma(gamma):
        bn_layer.update_gama(gamma)
        output = bn_layer.forward(input_tensor)
        return np.sum(output ** 2)
    
    computed_dx, computed_dgamma, computed_dbeta = bn_layer.batch_norm_backward(np.ones_like(input_tensor))
    numerical_dgamma = numerical_gradient(loss_fn_gamma, bn_layer.get_gama())
    
    error = np.linalg.norm(computed_dgamma - numerical_dgamma) / (np.linalg.norm(computed_dgamma) + 1e-7)
    print(f"Erro relativo nos gradientes do batch norm (gamma): {error}")
    assert error < 1e-3, "Os gradientes do batch normalization parecem incorretos!"
    print("Gradientes do batch normalization passaram no teste.")

# Teste para Adam
def test_adam_update(adam_optimizer, param_name, param, grad):
    """Testa a atualização do Adam verificando se segue a direção esperada."""
    updated_param = adam_optimizer.update(param_name, param, grad)
    assert np.all(updated_param != param), "Adam não está atualizando os parâmetros corretamente!"
    print("Otimizador Adam passou no teste.")
