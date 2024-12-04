from Neuron import Neuron
from App import App
from Chart import Chart
from initializations.He import He
from activations.Relu import Relu

def run_app():
    app: App = App.new_instance_with_model(
        model=Neuron({
            'input_size': 784, 
            'hidden_size':256, 
            'output_size':10, 
            "layers_number": 4,
            "learning_rate": 0.001,
            "regularization_lambda": 0.0001
        }, He()), 
        title="Reconhecimento de Dígitos"
    )
    app.model().set_activation(Relu())
    app.board().set_handler(app.predict_image)
    app.draw()
    app.train("mnist_train.csv", epochs=20, batch_size=32, plot=Chart().plot_activations)
    app.loop()

run_app()
