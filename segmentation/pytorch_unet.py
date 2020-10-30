### PyTorch UNet with Resnet 34 Backbone
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import statistics
import numpy as np
import os

def custom_transform(img):
    return torchvision.transforms.ToTensor(np.array(img))


### definition of PyTorch Lightning module in order to run everything
class PyTorch_UNet(pl.LightningModule):
    def __init__(self, config, classes, dataset='cityscapes'):
        super(PyTorch_UNet, self).__init__()
        self.config = config
        self.dataset = dataset
        self.model = smp.Unet('resnet34', encoder_weights=None, classes=classes)
        self.criterion = nn.CrossEntropyLoss()
        self.test_loss = None
        self.test_accuracy = None
        self.test_iou = None
        self.accuracy = pl.metrics.Accuracy()
        self.iou = pl.metrics.functional.classification.iou

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
        if self.dataset == 'cityscapes':
            return torch.utils.data.DataLoader(torchvision.datasets.Cityscapes(
                "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/", split='train', mode='fine', target_type='semantic',
                transform=torchvision.transforms.ToTensor(),
                target_transform=torchvision.transforms.ToTensor()),
                batch_size=int(self.config['batch_size']))
        else:
            # implement gis data
            pass

    def test_dataloader(self):
        if self.dataset == 'cityscapes':
            return torch.utils.data.DataLoader(
                torchvision.datasets.Cityscapes(
                    "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/", split='val', mode='fine', target_type='semantic',
                    transform=torchvision.transforms.ToTensor(),
                    target_transform=torchvision.transforms.ToTensor()),
                batch_size=int(self.config['batch_size']))
        else:
            # implement gis data
            pass

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
        return {'forward': self.forward(x), 'expected': y}

    def training_step_end(self, outputs):
        # only use when  on dp
        loss = self.criterion(outputs['forward'], outputs['expected'].long())
        logs = {'train_loss': loss}
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y}

    def test_step_end(self, outputs):
        loss = self.criterion(outputs['forward'], outputs['expected'].long())
        accuracy = self.accuracy(outputs['forward'], outputs['expected'])
        iou = self.iou(nn.LogSoftmax(outputs['forward']), outputs['expected'])
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

# load in gis data
def gis_dataloader():
    pass

def cityscapes_pt_objective(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    model = PyTorch_UNet(config, classes=30)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0, 1, 2, 3], distributed_backend='dp')
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model.model, model.test_iou

### two different objective functions, one for cityscapes and one for GIS

if __name__ == "__main__":
    # Note that batch size is per gpu
    test_config = {'batch_size': 8, 'learning_rate': .001, 'epochs': 1}
    res = cityscapes_pt_objective(test_config)
