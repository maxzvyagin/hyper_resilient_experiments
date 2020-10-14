### Tensorflow/Keras implementation of the AlexNext architectue for CIFAR100
import tensorflow as tf
from tensorflow import keras


class TensorFlow_AlexNet:
    def __init__(self, config):
        # get dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar100.load_data()
        # define the model using alexnet


        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        self.model.compile(optimizer=opt,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.config = config

    def fit(self):
        res = self.model.fit(self.x_train, self.y_train, epochs=self.config['epochs'],
                             batch_size=self.config['batch_size'])
        return res

    def test(self):
        res_test = self.model.evaluate(self.x_test, self.y_test)
        return res_test[1], self.model
