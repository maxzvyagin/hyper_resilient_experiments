### Tensorflow/Keras implementation of the AlexNext architectue for CIFAR100
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import numpy as np


class Fashion_TensorFlow_AlexNet:
    def __init__(self, config):
        tf.keras.backend.set_image_data_format('channels_first')
        tf.random.set_seed(0)
        b = int(config['batch_size'])
        # (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()
        f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/fashion_splits.pkl', 'rb')
        data = pickle.load(f)
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = data
        f.close()
        # transform = lambda i: np.expand_dims(i, -1)
        # self.x_train = transform(self.x_train)
        # self.x_val = transform(self.x_val)
        # self.x_test = transform(self.x_test)
        self.training_loss_history = None
        self.val_loss_history = None
        self.val_acc_history = None
        classes = 10
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', input_shape=(1, 28, 28),
                                padding="same"),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding="same"),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=1, activation='relu', padding="same"),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding="same"),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding="same"),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=1, activation='relu', padding="same"),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=1, activation='relu', padding="same"),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(config['dropout']),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(config['dropout']),
            keras.layers.Dense(classes, activation=None)
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=config['adam_epsilon'])
        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.config = config

    def fit(self):
        res = self.model.fit(self.x_train, self.y_train, epochs=self.config['epochs'],
                             batch_size=int(self.config['batch_size']), validation_data=(self.x_val, self.y_val),
                             shuffle=False)
        self.training_loss_history = res.history['loss']
        self.val_loss_history = res.history['val_loss']
        self.val_acc_history = res.history['val_accuracy']
        return res

    def test(self):
        res_test = self.model.evaluate(self.x_test, self.y_test)
        return res_test[1]

def fashion_tf_objective(config):
    model = Fashion_TensorFlow_AlexNet(config)
    model.fit()
    accuracy = model.test()
    # print(len(model.training_loss_history), len(model.val_loss_history), len(model.val_acc_history))
    return accuracy, model.model, model.training_loss_history, model.val_loss_history, model.val_acc_history


if __name__ == "__main__":
    test_config = {'batch_size': 500, 'learning_rate': .000001, 'epochs': 5, 'dropout': 0.9, 'adam_epsilon': 10**-9}
    res = fashion_tf_objective(test_config)