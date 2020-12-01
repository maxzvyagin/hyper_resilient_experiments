### PyTorch UNet with Resnet 34 Backbone
#import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import statistics
import numpy as np
import os
import sys
import argparse

from segmentation.gis_preprocess import pt_gis_train_test_split
from torch.utils.data import DataLoader

from segmentation.UNet.pytorch_unet import PyTorch_UNet_Model


def custom_transform(img):
    return torchvision.transforms.ToTensor(np.array(img))


### definition of PyTorch Lightning module in order to run everything
class PyTorch_UNet(pl.LightningModule):
    def __init__(self, config, classes, dataset='cityscapes', in_channels=3):
        super(PyTorch_UNet, self).__init__()
        self.config = config
        self.dataset = dataset
        # sigmoid is part of BCE with logits loss
        self.model = PyTorch_UNet_Model(in_channels, classes)
        if dataset == "gis":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.test_loss = None
        self.test_accuracy = None
        self.test_iou = None
        self.accuracy = pl.metrics.Accuracy()
        #self.iou = pl.metrics.functional.classification.iou
        if self.dataset == "gis":
            self.train_set, self.test_set = pt_gis_train_test_split()

    def train_dataloader(self):
        if self.dataset == 'cityscapes':
            return DataLoader(torchvision.datasets.Cityscapes(
                "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/", split='train', mode='fine', target_type='semantic',
                transform=torchvision.transforms.ToTensor(),
                target_transform=torchvision.transforms.ToTensor()),
                batch_size=int(self.config['batch_size']), num_workers=5)
            # return DataLoader(torchvision.datasets.Cityscapes(
            #     "/home/mzvyagin/datasets/", split='train', mode='fine', target_type='semantic',
            #     transform=torchvision.transforms.ToTensor(),
            #     target_transform=torchvision.transforms.ToTensor()),
            #     batch_size=int(self.config['batch_size']), num_workers=5)
        else:
            return DataLoader(self.train_set, batch_size=int(self.config['batch_size']), num_workers=10)

    def test_dataloader(self):
        if self.dataset == 'cityscapes':
            return DataLoader(torchvision.datasets.Cityscapes(
                "/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/", split='val', mode='fine', target_type='semantic',
                transform=torchvision.transforms.ToTensor(),
                target_transform=torchvision.transforms.ToTensor()),
                batch_size=int(self.config['batch_size']), num_workers=5)
            # return DataLoader(torchvision.datasets.Cityscapes(
            #     "/home/mzvyagin/datasets/", split='val', mode='fine', target_type='semantic',
            #     transform=torchvision.transforms.ToTensor(),
            #     target_transform=torchvision.transforms.ToTensor()),
            #     batch_size=int(self.config['batch_size']), num_workers=5)
        else:
            return DataLoader(self.test_set, batch_size=int(self.config['batch_size']), num_workers=5)

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
        if self.dataset == "gis":
            loss = self.criterion(outputs['forward'].squeeze(1), outputs['expected'])
        else:
            loss = self.criterion(outputs['forward'], outputs['expected'].long().squeeze(1))
        logs = {'train_loss': loss}
        return {'loss': loss, 'logs': logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        return {'forward': self.forward(x), 'expected': y}

    def test_step_end(self, outputs):
        if self.dataset == "gis":
            loss = self.criterion(outputs['forward'].squeeze(1), outputs['expected'])
            accuracy = self.accuracy(outputs['forward'].squeeze(1), outputs['expected'])
            # iou = self.iou(outputs['forward'].squeeze(1), outputs['expected'])
        else:
            loss = self.criterion(outputs['forward'], outputs['expected'].long().squeeze(1))
            accuracy = self.accuracy(outputs['forward'], outputs['expected'].squeeze(1))
            # iou = self.iou(outputs['forward'], outputs['expected'].squeeze(1))
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
        # iou = []
        # for x in outputs:
        #     iou.append(float(x['test_iou']))
        # avg_iou = statistics.mean(iou)
        # self.test_iou = avg_iou
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_accuracy': avg_accuracy}


def segmentation_pt_objective(config, dataset="cityscapes"):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.manual_seed(0)
    if dataset == "cityscapes":
        model = PyTorch_UNet(config, classes=30)
    else:
        model = PyTorch_UNet(config, classes=1, dataset=dataset, in_channels=4)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=[0, 1, 2, 3, 4, 5, 6, 7], distributed_backend='dp')
    trainer.fit(model)
    trainer.test(model)
    return model.test_accuracy, model.model


def cityscapes_pt_objective(config):
    return segmentation_pt_objective(config, dataset="cityscapes")


def gis_pt_objective(config):
    return segmentation_pt_objective(config, dataset="gis")


### two different objective functions, one for cityscapes and one for GIS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch')
    args = parser.parse_args()
    if args.batch:
        batch_size = args.batch
    else:
        batch_size = 4
    test_config = {'batch_size': 16, 'learning_rate': .001, 'epochs': 1}
    #res = segmentation_pt_objective(test_config)
    res = segmentation_pt_objective(test_config, dataset="gis")
