import argparse
import neural_network as nn
import neural_network.supply as attr

from neural_network import Config
from neural_network.board import Chart, FileInput
from neural_network.core.padding import Padding
from neural_network.core.image_processor import ImageProcessor
from neural_network.console import Commands
from custom.residual_block import ResidualBlock

IMAGE_SIZE = (100, 100)
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 50

def create_configuration():
    config = Config({
        'input_shape': (IMAGE_CHANNELS, *IMAGE_SIZE),
        'regularization_lambda': 1e-4,
    })

    config.set_processor(
        ImageProcessor(
            base_dir="./data/breast-histopathology-images",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            split_ratios=(0.7, 0.15, 0.15),
            shuffle=True,
            augmentation=True,
            augmentation_params={
                'rotation': 30,
                'zoom': 0.2,
                'blur': True
            }
        )
    )

    config.driver('cpu')
    config.set_global_optimizer(attr.Adam(learning_rate=0.0005))
    config.with_cache(path='./data/cache/model.pkl')
    config.padding_type(Padding.SAME)
    config.loss_function(attr.CrossEntropyLoss())

    # Camadas iniciais
    kernel1 = config.add_kernel(number=64, shape=(7, 7), stride=2) 
    kernel1.initializer(attr.He())
    kernel1.activation(attr.LeakyRelu(alpha=0.01))
    kernel1.batch_normalization()
    kernel1.max_pooling(shape=(2, 2), stride=2)

    # **ResNet**
    for filters, stride in [(64, 1), (128, 2), (256, 1), (512, 2)]:
        config.add_custom(ResidualBlock(number=filters, shape=(3, 3), stride=stride, downsample=True))
        config.add_custom(ResidualBlock(number=filters, shape=(3, 3), stride=1, downsample=False))

    # **Flatten**
    config.flatten()
    dense = config.dense()

    # **Fully Connected Layers**
    layer1 = dense.add_layer(size=1024, dropout=0.5)
    layer1.initializer(attr.He())
    layer1.activation(attr.LeakyRelu(alpha=0.01))

    layer2 = dense.add_layer(size=512, dropout=0.5)
    layer2.initializer(attr.He())
    layer2.activation(attr.LeakyRelu(alpha=0.01))

    layer3 = dense.add_layer(size=256, dropout=0.5)
    layer3.initializer(attr.He())
    layer3.activation(attr.LeakyRelu(alpha=0.01))

    # **Saída**
    output = dense.add_layer(size=2)
    output.activation(attr.Softmax())
    output.initializer(attr.Xavier())

    return config

def create_app(config: Config) -> nn.App: 
    return nn.app.App(
        model=config.new_model(), 
        board=FileInput(
            title="Breast Cancer Recognition",
            img_resize=IMAGE_SIZE,
        )
    )

def train_model(app: nn.App, plot: bool):
    app.model().set_training_mode()
    app.model().get_trainer().train(
        epochs=EPOCHS,
        plot=Chart().plot_metrics if plot else None,
        scheduler=attr.ReduceLROnPlateau(factor=0.2, patience=4, min_lr=1e-6)
    )

def validate_model(app: nn.App):
    app.draw()
    app.model().set_test_mode()
    app.board().set_handler(handler=app.predict_image)
    app.board().set_labels(path_json="./labels/breast_cancer.json")
    app.loop()

def test_model(app: nn.App, plot: bool):
    app.model().set_test_mode()
    app.model().get_tester().test(plot=Chart().plot_metrics if plot else None)

def main():
    commands = Commands(argparse.ArgumentParser(description="Train or test the model"))
    commands.register()

    args = commands.get_args()
    config = create_configuration()

    if args.clear_cache: 
        config.restore_initialization_cache()

    if args.no_cache: 
        config.with_no_cache() 

    app = create_app(config)

    modes = {
        "train": lambda: train_model(app, args.plot),
        "validate": lambda: validate_model(app),
        "test": lambda: test_model(app, args.plot),
    }

    action = modes.get(args.mode)
    action()

if __name__ == "__main__":
    main()
