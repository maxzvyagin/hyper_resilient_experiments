### Tensorflow/Keras implementation of the AlexNext architectue for CIFAR100
import tensorflow as tf
from tensorflow import keras
import os
import tensorflow_datasets as tfds

class Fashion_TensorFlow_AlexNet:
    def __init__(self, config):
        tf.keras.backend.set_image_data_format('channels_last')
        tf.random.set_seed(0)
        b = int(config['batch_size'])
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.caltech101.load_data()
        # f = lambda i: tf.expand_dims(i, -1)
        # self.x_train = f(self.x_train)
        # self.x_test = f(self.x_test)
        classes = 101
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', input_shape=(28, 28, 1)),
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

def get_cityscapes():
    """ Returns test, train split of Cityscapes data"""
    # first try loading from cache object, otherwise load from scratch

    train, test = tfds.load('caltech101', split=['train', 'test'], shuffle_files=False)
    # train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False,
    #                         data_dir='/home/mzvyagin/datasets/')
    train = list(train)
    train_x = [pair['image_left'] for pair in train]
    train_y = [pair['segmentation_label'] for pair in train]
    train_x = list(map(lambda x: x.numpy() / 255.0, train_x))
    # train_x, train_y = [], []
    # for i in train:
    #     train_x.append(i['image_left'].numpy() / 255)
    #     train_y.append(i['segmentation_label'].numpy() / 255)
    # test_x, test_y = [], []
    test = list(test)
    test_x = [pair['image_left'] for pair in test]
    test_y = [pair['segmentation_label'] for pair in test]
    train_x = list(map(lambda x: tf.convert_to_tensor(x.numpy() / 255.0), test_x))
    # for i in test:
    #     test_x.append(i['image_left'].numpy() / 255)
    #     test_y.append(i['segmentation_label'].numpy() / 255)
    return (train_x, train_y), (test_x, test_y)

if __name__ == "__main__":
    test_config = {'batch_size': 64, 'learning_rate': .000001, 'epochs': 100, 'dropout': 0.5, 'adam_epsilon': 10**-9}
    res = fashion_tf_objective(test_config)