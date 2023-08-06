import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.optim as optim
import argparse
import logging
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


class SimpleDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self._img_labels = pd.read_csv(annotations_file)
        self._transform = transform
        self._target_transform = target_transform
    
    def __len__(self):
        return len(self._img_labels)
    
    def __getitem__(self, idx):
        img_path = self._img_labels.iloc[idx,0]
        image = read_image(img_path)
        label = self._img_labels.iloc[idx,1]
        if self._transform:
            image = self._transform(image)
        if self._target_transform:
            label = self._target_transform(label)
        
        return image, label        

class SimpleCNN():
    def __init__(self):
        self.n_epochs = 0
        self.lr = 0

    def _find_device(self):
        device = (
            'cuda' if torch.cuda.is_available() else 'cpu'
            )
        return device

    def load_data(self, batch_size):
        train_dataset = SimpleDataset('../data/indexes/index_train.csv')
        test_dataset = SimpleDataset('../data/indexes/index_test.csv')
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return (train_loader, test_loader)

    def create_model(self, out_classes):

        model = nn.Sequential()
        model.add_module('Conv2D_1', nn.Conv2d(1, 32, kernel_size=(2,2)))
        model.add_module('ReLU', nn.ReLU())
        model.add_module('MaxPool', nn.MaxPool2d(kernel_size=(2,2)))
        model.add_module('Flatten', nn.Flatten())
        model.add_module('LazyLinear', nn.LazyLinear(out_features=out_classes))

        return model

    def train_model(self):
        pass

def go(args):

    
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
