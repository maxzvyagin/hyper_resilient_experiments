### Tensorflow/Keras implementation of the AlexNext architectue for CIFAR100
import tensorflow as tf
from tensorflow import keras
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class TensorFlow_AlexNet:
    def __init__(self, config):
        tf.random.set_seed(0)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar100.load_data()
        # define the model using alexnet architecture
        # from: https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
        # updated to match existing pytorch model
        #os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[4:8], 'GPU')
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])
        with strategy.scope():
            self.model = keras.models.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(11,11), strides=4, activation='relu', input_shape=(32, 32, 3)),
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
                keras.layers.Dense(100, activation=tf.nn.log_softmax)
            ])

            opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
            self.model.compile(optimizer=opt,
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.config = config

    def fit(self):
        res = self.model.fit(self.x_train, self.y_train, epochs=self.config['epochs'],
                             batch_size=int(self.config['batch_size']))
        return res

    def test(self):
        res_test = self.model.evaluate(self.x_test, self.y_test)
        return res_test[1]

def cifar_tf_objective(config):
    # gpu_config = ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=gpu_config)
    model = TensorFlow_AlexNet(config)
    model.fit()
    accuracy = model.test()
    return accuracy, model.model

if __name__ == "__main__":
    test_config = {'batch_size': 64, 'learning_rate': .001, 'epochs': 1, 'dropout': 0.5}
    res = cifar_tf_objective(test_config)