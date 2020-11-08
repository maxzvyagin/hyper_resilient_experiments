### Tensorflow UNet with Resnet34 Backbone
import segmentation_models as sm
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys

sys.path.append("/home/mzvyagin/hyper_resilient/segmentation")
from gis_preprocess import tf_gis_test_train_split


def cityscapes_tf_objective(config, classes=30):
    tf.random.set_seed(0)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])
    with strategy.scope():
        model = tf.keras.Sequential()
        model.add(sm.Unet('resnet34', encoder_weights=None, classes=classes, activation=None))
        model.add(tf.keras.layers.Dense(30, activation=tf.nn.log_softmax))
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
    # fit model on cityscapes data
    (x_train, y_train), (x_test, y_test) = get_cityscapes()
    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    res_test = model.evaluate(x_test, y_test)
    return res_test[1], model


# same model just using gis data instead
def gis_tf_objective(config, classes=1):
    tf.random.set_seed(0)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])
    with strategy.scope():
        model = sm.Unet('resnet34', encoder_weights=None, classes=classes, input_shape=(None, None, 4),
                        activation="sigmoid")
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])
    # fit model on gis data
    (x_train, y_train), (x_test, y_test) = tf_gis_test_train_split()
    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=int(config['batch_size']))
    res_test = model.evaluate(x_test, y_test)
    return res_test[1], model


def get_cityscapes():
    """ Returns test, train split of Cityscapes data"""
    # train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False,
    #                         data_dir='/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/')
    train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False,
                            data_dir='/home/mzvyagin/datasets/')
    train = list(train)
    train_x, train_y = [], []
    for i in train:
        train_x.append(i['image_left'].numpy() / 255)
        train_y.append(i['segmentation_label'].numpy() / 255)
    test_x, test_y = [], []
    test = list(test)
    for i in test:
        test_x.append(i['image_left'].numpy() / 255)
        test_y.append(i['segmentation_label'].numpy() / 255)
    return (train_x, train_y), (test_x, test_y)


if __name__ == "__main__":
    test_config = {'batch_size': 16, 'learning_rate': .001, 'epochs': 1}
    res = cityscapes_tf_objective(test_config)
    # print(res[0])
    res = gis_tf_objective(test_config)
