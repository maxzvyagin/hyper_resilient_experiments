from torch import nn
import pytorch_lightning as pl
import torchvision
import torch
import statistics
import os


class NumberNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, 10))
        ## nn.Softmax())
        # not including softmax because it's included in the Cross Entropy Loss Function
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.test_loss = None
        self.test_accuracy = None
        self.accuracy = pl.metrics.Accuracy()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=True,
                                                                      transform=torchvision.transforms.ToTensor(),
                                                                      target_transform=None, download=True),
                                           batch_size=int(self.config['batch_size']), num_workers=5)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=False,
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


def mnist_pt_objective(config):
    torch.manual_seed(0)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    model = NumberNet(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0])
    # trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0], distributed_backend='ddp')
    # trainer = pl.Trainer(max_epochs=config['epochs'], gpus=4, auto_select_gpus=True)
    # trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[8, 9, 10, 11], distributed_backend='ddp')
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model.model


if __name__ == "__main__":
    test_config = {'batch_size': 64, 'learning_rate': .001, 'epochs': 1, 'dropout': 0.5}
    res = mnist_pt_objective(test_config)
