# import tensorflow as tf
from torch import nn
import pytorch_lightning as pl
from tqdm import tqdm
import torch
import torchvision
import statistics
import pandas as pd
import foolbox as fb
import tensorflow as tf
import scipy
import pickle
from concurrent.futures import ProcessPoolExecutor
import concurrent


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
            # not include softmax because it's included in the Cross Entropy Loss Function
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.test_loss = None
        self.test_accuracy = None
        self.accuracy = pl.metrics.Accuracy()


    def train_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=True,
                                                                      transform=torchvision.transforms.ToTensor(),
                                                                      target_transform=None, download=True),
                                           batch_size=int(self.config['batch_size']))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(torchvision.datasets.MNIST("~/resiliency/", train=True,
                                                                      transform=torchvision.transforms.ToTensor(),
                                                                      target_transform=None, download=True),
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


def mnist_pt_objective(config):
    model = NumberNet(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], gpus=1, auto_select_gpus=True)
    trainer.fit(model)
    trainer.test(model)
    pt_model_weights = list(model.parameters())
    just_pt_weights = list()
    for w in pt_model_weights:
        just_pt_weights.extend(w.cpu().detach().numpy().flatten())
    return just_pt_weights, config


def mnist_tf_objective(config):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(config['dropout']),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    res_test = model.evaluate(x_test, y_test)
    just_tf_weights = list()
    # get weights
    for w in model.weights:
        just_tf_weights.extend(w.numpy().flatten())
    return just_tf_weights, config

if __name__ == "__main__":
    top_config

    top_pt, top_tf, bottom_pt, bottom_tf = [], [], [], []
    for i in range(50):
        top_pt_accuracy, top_pt_model = mnist_pt_objective(top_config)
        top_tf_accuracy, top_tf_model = mnist_tf_objective(top_config)
        bottom_pt_accuracy, bottom_pt_model = mnist_pt_objective(bottom_config)
        bottom_tf_accuracy, bottom_tf_model = mnist_tf_objective(bottom_config)
        # get the output layer weights
        top_pt_out = list(top_pt_model.parameters())[-1].detach().numpy()
        top_pt.append(top_pt_out)
        top_tf_out = top_tf_model.weights[-1].numpy()
        top_tf.append(top_tf_out)
        bottom_pt_out = list(bottom_pt_model.parameters())[-1].detach().numpy()
        bottom_pt.append(bottom_pt_out)
        bottom_tf_out = bottom_tf_model.weights[-1].numpy()
        bottom_tf.append(bottom_tf_out)
