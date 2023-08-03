import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
import argparse
import logging
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


class SimpleCNN():
    def __init__(self):
        self.n_epochs = 0
        self.lr = 0

    def _find_device(self):
        device = (
            'cuda' if torch.cuda.is_available() else 'cpu'
            )
        return device

    def load_data(self):
        train_dataset = DataLoader()
        test_dataset = DataLoader()
        return (train_dataset, test_dataset)

    def create_model(self):
        pass

    def train_model(self):
        pass

def go(args):

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
print(f"Using {device} device")
    
    ## load the train dataset
    
    ## define a way to dinamically define the model architecture
    ## and train it
    
    ## log the training into mlflow
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Split dataset, please provide the image input path, \
        image extension to be looked for, the test size (in pct) and the split random state.')
    
    parser.add_argument(
        "--img_input_path",
        type = str,
        help = "Path where we can find the images.",
        required= True
    )
    
    parser.add_argument(
        "--img_extension",
        type = str,
        help = "Image extension to be used.",
        required= True
    )
    
    parser.add_argument(
        "--test_pct",
        type = float,
        help = "Test size in percentage.",
        required= True
    )
    
    parser.add_argument(
        "--split_random_state",
        type = int,
        help = "Sklearn's train_test_split random_state paramenter.",
        required= True
    )
    
    args = parser.parse_args()
    
    go(args)
