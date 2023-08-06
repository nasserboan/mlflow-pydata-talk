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
from mlflow.models import infer_signature

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
    def __init__(self, n_epochs, lr):
        self.n_epochs = n_epochs
        self.lr = lr

    def _find_device(self):
        device = (
            'cuda' if torch.cuda.is_available() else 'cpu'
            )
        return device

    def load_data(self, batch_size):
        train_dataset = SimpleDataset('../data/indexes/index_train.csv')
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        return train_loader

    def create_model(self, out_classes):

        model = nn.Sequential()
        model.add_module('Conv2D_1', nn.Conv2d(1, 32, kernel_size=(2,2)))
        model.add_module('ReLU', nn.ReLU())
        model.add_module('MaxPool', nn.MaxPool2d(kernel_size=(2,2)))
        model.add_module('Flatten', nn.Flatten())
        model.add_module('LazyLinear', nn.LazyLinear(out_features=out_classes))

        return model

    def train_model(self, model_to_train, train_data):
        train_steps = len(train_data) // train_data.batch_size
        opt = optim.SGD(model_to_train.parameters(), lr= self.lr)
        loss_fn = nn.CrossEntropyLoss()

        H = {
            'train_loss':[],
            'train_acc':[]
        }

        for e in range(0,self.n_epochs):
            model_to_train.train()

            total_train_loss = 0
            total_train_correct = 0

            for (X, y) in train_data:
                (X, y) = (X.to(self._find_device), y.to(self._find_device))

                pred = model_to_train(X)
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_train_loss += loss
                total_train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            H['train_loss'].append(total_train_loss/train_steps)
            H['train_acc'].append(total_train_correct/len(train_data.dataset))
        
        signature = infer_signature(X.numpy(), model_to_train(X).detach().numpy())

        return model_to_train, H, signature

def go(args):

    cnn_helper = SimpleCNN(args.epochs, args.lr)
    train_loader = cnn_helper.load_data(batch_size=args.batch_size)
    raw_model = cnn_helper.create_model(args.out_classes)
    trained_model, H, sig = cnn_helper.train_model(raw_model, train_loader)

    mlflow.pytorch.log_model('model', trained_model, signature=sig)
    mlflow.log_dict(H)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Split dataset, please provide the image input path, \
        image extension to be looked for, the test size (in pct) and the split random state.')
    
    parser.add_argument(
        "--img_input_path",
        type = str,
        help = "Path where we can find the images.",
        required= True
    )
    
    args = parser.parse_args()
    
    go(args)
