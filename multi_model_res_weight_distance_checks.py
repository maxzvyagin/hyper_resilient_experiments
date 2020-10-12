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
    low_config_list = [{'learning_rate': 0.09949307671494452,
                              'dropout': 0.8847296070468049,
                              'epochs': 68,
                              'batch_size': 67},
                             {'learning_rate': 0.08694568938709626,
                              'dropout': 0.8930164363399653,
                              'epochs': 85,
                              'batch_size': 604},
                             {'learning_rate': 0.060202546820720335,
                              'dropout': 0.8453783367544603,
                              'epochs': 47,
                              'batch_size': 72},
                             {'learning_rate': 0.08218697437893893,
                              'dropout': 0.8996455090169381,
                              'epochs': 49,
                              'batch_size': 100},
                             {'learning_rate': 0.0886164883077212,
                              'dropout': 0.8469608869505587,
                              'epochs': 95,
                              'batch_size': 60}]

    high_config_list = [{'learning_rate': 0.0008247026118279462,
                              'dropout': 0.2465723148341217,
                              'epochs': 48,
                              'batch_size': 380},
                             {'learning_rate': 0.012400983319850551,
                              'dropout': 0.2753653256155508,
                              'epochs': 66,
                              'batch_size': 903},
                             {'learning_rate': 0.00629794626194513,
                              'dropout': 0.20527720389732526,
                              'epochs': 63,
                              'batch_size': 425},
                             {'learning_rate': 0.011433179648786712,
                              'dropout': 0.33713250206233825,
                              'epochs': 54,
                              'batch_size': 834},
                             {'learning_rate': 0.008332646839818986,
                              'dropout': 0.3985947224532251,
                              'epochs': 64,
                              'batch_size': 950}]
    ### high models
    high_pt_models = {}
    high_tf_models = {}
    pt_workers = []
    tf_workers = []
    with ProcessPoolExecutor(max_workers=50) as e:
        for config in high_config_list:
            pt = e.submit(mnist_pt_objective, config)
            pt_workers.append(pt)
            tf = e.submit(mnist_tf_objective, config)
            tf_workers.append(tf)

    for result in concurrent.futures.as_completed(tf_workers):
        weights, config = result.result()
        config_name = "{}lr_{}drop_{}epochs_{}batch".format(config['learning_rate'], config['dropout'],
                                                            config['epochs'], config['batch_size'])
        high_tf_models[config_name] = weights

    for result in concurrent.futures.as_completed(pt_workers):
        weights, config = result.result()
        config_name = "{}lr_{}drop_{}epochs_{}batch".format(config['learning_rate'], config['dropout'],
                                                            config['epochs'], config['batch_size'])
        high_pt_models[config_name] = weights

    with open("top_5_config_pt_model_weights.pkl", "wb") as f:
        pickle.dump(high_pt_models, f)
    with open("top_5_config_tf_model_weights.pkl", "wb") as f:
        pickle.dump(high_tf_models, f)


    ### low models
    low_pt_models = {}
    low_tf_models = {}
    pt_workers = []
    tf_workers = []
    with ProcessPoolExecutor(max_workers=50) as e:
        for config in low_config_list:
            pt = e.submit(mnist_pt_objective, config)
            pt_workers.append(pt)
            tf = e.submit(mnist_tf_objective, config)
            tf_workers.append(tf)

    for result in concurrent.futures.as_completed(tf_workers):
        weights, config = result.result()
        config_name = "{}lr_{}drop_{}epochs_{}batch".format(config['learning_rate'], config['dropout'],
                                                            config['epochs'], config['batch_size'])

        low_tf_models[config_name] = weights

    for result in concurrent.futures.as_completed(pt_workers):
        weights, config = result.result()
        config_name = "{}lr_{}drop_{}epochs_{}batch".format(config['learning_rate'], config['dropout'],
                                                            config['epochs'], config['batch_size'])
        low_pt_models[config_name] = weights

    with open("bottom_5_config_pt_model_weights.pkl", "wb") as f:
        pickle.dump(low_pt_models, f)
    with open("bottom_5_config_tf_model_weights.pkl", "wb") as f:
        pickle.dump(low_tf_models, f)
