from hyperspace import create_hyperspace
from ray import tune
#import tensorflow as tf
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from ray.tune.suggest.skopt import SkOptSearch
from skopt import Optimizer
import ray
from tqdm import tqdm
import torch
import torchvision
import statistics
import pandas as pd
import foolbox as fb


### Defining the hyperspace
hyperparameters = [(0.00001, 0.1),  # learning_rate
                   (0.2, 0.9),  # dropout
                   (10, 100),  # epochs 
                   (10, 1000)]  # batch size
space = create_hyperspace(hyperparameters)

class NumberNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(784, 128), 
            nn.ReLU(), 
            nn.Dropout(config['dropout']), 
            nn.Linear(128, 10), 
            nn.Softmax())
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.test_loss = None
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=True, 
                                                                      transform=torchvision.transforms.ToTensor(), target_transform=None, download=True), 
                                           batch_size=int(self.config['batch_size']))
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=True, 
                                                                      transform=torchvision.transforms.ToTensor(), target_transform=None, download=True), 
                                           batch_size=int(self.config['batch_size']))
    
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
        return {'loss': loss}
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        logs = {'test_loss': loss}
        return {'test_loss': loss, 'logs': logs}
    
    def test_epoch_end(self, outputs):
        loss = []
        for x in outputs:
            loss.append(float(x['test_loss']))
        avg_loss = statistics.mean(loss)
        tensorboard_logs = {'test_loss': avg_loss}
        self.test_loss = avg_loss
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


def mnist_pt_objective(config):
    model = NumberNet(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, auto_select_gpus=True)
    trainer.fit(model)
    trainer.test(model)
    tune.report(test_loss=model.test_loss)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=config['batch_size'])
    clean_accuracy = fb.utils.accuracy(fmodel, images, labels)
    attack = fb.attacks.SaltAndPepperNoiseAttack()
    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    robust_accuracy = 1 - success.cpu().numpy().astype(float).flatten().mean(axis=-1)
    # res test[0] reports the loss from the evaluation, res_test[1] reports the accuracy
    tune.report(robust_acc = robust_accuracy)
    return robust_accuracy

if __name__=="__main__":
        results = []
        for section in tqdm(space):
            # create a skopt gp minimize object
            optimizer = Optimizer(section)
            search_algo = SkOptSearch(optimizer, ['learning_rate', 'dropout', 'epochs', 'batch_size'],
                                      metric='robust_acc', mode='max')
            # not using a gpu because running on local
            analysis = tune.run(mnist_pt_objective, search_alg=search_algo, num_samples=20, resources_per_trial={'gpu':1})
            results.append(analysis)

        print(type(results[0]))

        all_pt_results = results[0].results_df
        for i in range(1, len(results)):
            all_pt_results = all_pt_results.append(results[i].results_df)

        all_pt_results.to_csv('pt_fool_saltandpepper.csv')
