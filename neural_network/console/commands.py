import argparse

class Commands:
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser
        
    def register(self):
        self.parser.add_argument(
            "--mode", 
            choices=["train", "test"], 
            default="test",
            help="Choose 'train' to train the model or 'test' to test the model."
        )
        
        self.parser.add_argument(
            "--clear-cache", 
            action="store_true",
            help="Clears the cache of the convolutional network, including stored weights and configurations."
        )
        
        self.parser.add_argument(
            "--no-cache", 
            action="store_true",
            help="Uses new weights and restores the defined configurations."
        )

        self.parser.add_argument(
            "--plot", 
            action="store_true",
            default=False,
            help="Plot loss and accuracy into graphics during the model training."
        )
        
    def get_args(self):
        return self.parser.parse_args()