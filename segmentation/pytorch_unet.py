### PyTorch UNet with Resnet 34 Backbone
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import statistics
import numpy as np

def custom_transform(img):
    return torchvision.transforms.ToTensor(np.array(img))


### definition of PyTorch Lightning module in order to run everything
class PyTorch_UNet(pl.LightningModule):
    def __init__(self, config, classes=20):
        super(PyTorch_UNet, self).__init__()
        self.config = config
        self.model = smp.Unet('resnet34', encoder_weights=None, classes=classes)
        self.criterion = nn.CrossEntropyLoss()
        self.test_loss = None
        self.test_accuracy = None
        self.accuracy = pl.metrics.Accuracy()

    # def train_dataloader(self):
    #     return torch.utils.data.DataLoader(torchvision.datasets.Cityscapes(
    #         "~/datasets/pytorch/", split='train', mode='coarse', target_type='semantic',
    #         transform=torchvision.transforms.ToTensor(),
    #         target_transform=torchvision.transforms.ToTensor()),
    #         batch_size=int(self.config['batch_size']))
    #
    # def test_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #         torchvision.datasets.Cityscapes(
    #             "~/datasets/pytorch", split='val', mode='coarse', target_type='semantic',
    #             transform=torchvision.transforms.ToTensor(),
    #             target_transform=torchvision.transforms.ToTensor()),
    #         batch_size=int(self.config['batch_size']))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.VOCSegmentation("~/datasets/pytorch/", download=True))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.VOCSegmentation("~/datasets/pytorch/", download=True,
                                                                                image_set="val"))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)
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


def cityscapes_pt_objective(config):
    model = PyTorch_UNet(config, classes=20)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, auto_select_gpus=True)
    #trainer = pl.Trainer(max_epochs=config['epochs'])
    #trainer.fit(model)
    #trainer.test(model)
    #return model.test_accuracy, model.model
    return model.model


### two different objective functions, one for cityscapes and one for GIS

if __name__ == "__main__":
    test_config = {'batch_size': 64, 'learning_rate': .001, 'epochs': 1}
    res = cityscapes_pt_objective(test_config)
