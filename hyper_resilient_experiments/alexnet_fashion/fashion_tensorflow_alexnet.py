### Tensorflow/Keras implementation of the AlexNext architectue for CIFAR100
import tensorflow as tf
from tensorflow import keras
import os


class Fashion_TensorFlow_AlexNet:
    def __init__(self, config):
        tf.keras.backend.set_image_data_format('channels_last')
        tf.random.set_seed(0)
        b = int(config['batch_size'])
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()
        classes = 10
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', input_shape=(28, 28)),
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
                             batch_size=int(self.config['batch_size']))
        return res

    def test(self):
        res_test = self.model.evaluate(self.x_test, self.y_test)
        return res_test[1]

def fashion_tf_objective(config):
    model = Fashion_TensorFlow_AlexNet(config)
    model.fit()
    accuracy = model.test()
    return accuracy, model.model


if __name__ == "__main__":
    test_config = {'batch_size': 500, 'learning_rate': .000001, 'epochs': 20, 'dropout': 0.9, 'adam_epsilon': 10**-9}
    res = fashion_tf_objective(test_config)