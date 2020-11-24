import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import statistics
import os
import argparse


class PyTorch_AlexNet(pl.LightningModule):
    def __init__(self, config, classes=100, ten=False):
        super(PyTorch_AlexNet, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(4096, classes))
        self.criterion = nn.CrossEntropyLoss()
        self.test_loss = None
        self.test_accuracy = None
        self.accuracy = pl.metrics.Accuracy()
        self.ten = ten

    def train_dataloader(self):
        if self.ten:
            return DataLoader(torchvision.datasets.CIFAR10("~/datasets/", train=True,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(self.config['batch_size']), num_workers=5)
        else:
            return DataLoader(torchvision.datasets.CIFAR100("~/datasets/", train=True,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(self.config['batch_size']), num_workers=5)

    def test_dataloader(self):
        if self.ten:
            return DataLoader(torchvision.datasets.CIFAR10("~/datasets/", train=False,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(self.config['batch_size']), num_workers=5)
        else:
            return DataLoader(torchvision.datasets.CIFAR100("~/datasets/", train=False,
                                                            transform=torchvision.transforms.ToTensor(),
                                                            target_transform=None, download=True),
                              batch_size=int(self.config['batch_size']), num_workers=5)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=self.config['adam_epsilon'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        return {'forward': self.forward(x), 'expected': y}

    def training_step_end(self, outputs):
        # only use when  on dp
        loss = self.criterion(outputs['forward'], outputs['expected'])
        logs = {'train_loss': loss}
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y}

    def test_step_end(self, outputs):
        loss = self.criterion(outputs['forward'], outputs['expected'])
        accuracy = self.accuracy(outputs['forward'], outputs['expected'])
        logs = {'test_loss': loss, 'test_accuracy': accuracy}
        return {'test_loss': loss, 'logs': logs, 'test_accuracy': accuracy}

    def test_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['test_loss']))
        avg_loss = statistics.mean(loss)
        tensorboard_logs = {'test_loss': avg_loss}
        self.test_loss = avg_loss
        accuracy = []
        for x in outputs:
            accuracy.append(float(x['test_accuracy']))
        avg_accuracy = statistics.mean(accuracy)
        self.test_accuracy = avg_accuracy
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_accuracy': avg_accuracy}


def cifar100_pt_objective(config, ten=False):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.manual_seed(0)
    model = PyTorch_AlexNet(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0])
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model.model

def cifar10_pt_objective(config):
    torch.manual_seed(0)
    model = PyTorch_AlexNet(config, classes=10, ten=True)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0])
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model.model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch")
    args = parser.parse_args()
    if args.batch:
        batch = args.batch
    else:
        batch = 64
    test_config = {'batch_size': batch, 'learning_rate': .001, 'epochs': 1, 'dropout': 0.5}
    res = cifar100_pt_objective(test_config)
