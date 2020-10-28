### PyTorch UNet with Resnet 34 Backbone
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import statistics
import numpy as np
import sklearn.metrics

def custom_transform(img):
    return torchvision.transforms.ToTensor(np.array(img))


### definition of PyTorch Lightning module in order to run everything
class PyTorch_UNet(pl.LightningModule):
    def __init__(self, config, classes):
        super(PyTorch_UNet, self).__init__()
        self.config = config
        self.model = smp.Unet('resnet34', encoder_weights=None, classes=classes)
        self.criterion = nn.CrossEntropyLoss()
        self.test_loss = None
        self.test_accuracy = None
        self.test_iou = None
        self.accuracy = pl.metrics.Accuracy()
        self.iou = sklearn.metrics.jaccard_score

    # def train_dataloader(self):
    #     train = torchvision.datasets.CocoDetection(
    #         '~/datasets/coco/train2017', '~/datasets/coco/annotations/instances_train2017.json',
    #         transform=torchvision.transforms.ToTensor(), target_transform=torchvision.transforms.ToTensor())
    #     return torch.utils.data.DataLoader(train, batch_size=int(self.config['batch_size']))
    #
    # def test_dataloader(self):
    #     test = torchvision.datasets.CocoDetection(
    #         '~/datasets/coco/val2017', '~/datasets/coco/annotations/instances_val2017.json',
    #         transform=torchvision.transforms.ToTensor(), target_transform=torchvision.transforms.ToTensor())
    #     return torch.utils.data.DataLoader(test, batch_size=int(self.config['batch_size']))


    def train_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.Cityscapes(
            "~/datasets/", split='train', mode='fine', target_type='semantic',
            transform=torchvision.transforms.ToTensor(),
            target_transform=torchvision.transforms.ToTensor()),
            batch_size=int(self.config['batch_size']))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.Cityscapes(
                "~/datasets/", split='val', mode='fine', target_type='semantic',
                transform=torchvision.transforms.ToTensor(),
                target_transform=torchvision.transforms.ToTensor()),
            batch_size=int(self.config['batch_size']))

    # def train_dataloader(self):
    #     return torch.utils.data.DataLoader(torchvision.datasets.VOCSegmentation("~/datasets/pytorch/", download=True))
    #
    # def test_dataloader(self):
    #     return torch.utils.data.DataLoader(torchvision.datasets.VOCSegmentation("~/datasets/pytorch/", download=True,
    #                                                                             image_set="val"))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, torch.squeeze(y.long(), 1))
        logs = {'train_loss': loss}
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        y = y.long()
        loss = self.criterion(logits, torch.squeeze(y, 1))
        accuracy = self.accuracy(logits, torch.squeeze(y, 1))
        iou = self.iou(logits, torch.squeeze(y, 1))
        logs = {'test_loss': loss, 'test_accuracy': accuracy, 'test_iou': iou}
        return {'test_loss': loss, 'logs': logs, 'test_accuracy': accuracy, 'test_iou': iou}

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
        iou = []
        for x in outputs:
            iou.append(float(x['test_iou']))
        avg_iou = statistics.mean(iou)
        self.test_iou = avg_iou
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_accuracy': avg_accuracy}


def cityscapes_pt_objective(config):
    model = PyTorch_UNet(config, classes=30)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, auto_select_gpus=True, distributed_backend='ddp')
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model.model, model.test_iou


### two different objective functions, one for cityscapes and one for GIS

if __name__ == "__main__":
    #batch size is per gpu
    test_config = {'batch_size': 3, 'learning_rate': .001, 'epochs': 1}
    res = cityscapes_pt_objective(test_config)
