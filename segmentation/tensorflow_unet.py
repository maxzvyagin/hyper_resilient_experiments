### Tensorflow UNet with Resnet34 Backbone
import segmentation_models as sm
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys
from tensorflow import keras

from segmentation.gis_preprocess import tf_gis_test_train_split

# from segmentation.UNet.tensorflow_unet import TensorFlow_UNet_Model
# from model import unet

# import os


def cityscapes_tf_objective(config, classes=30):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    b = int(config['batch_size'])
    tf.random.set_seed(0)
    keras.backend.set_image_data_format('channels_last')
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[4:8], 'GPU')
    # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5",
    #                                                    "/gpu:6", "/gpu:7"])
    # with strategy.scope():
    model = tf.keras.Sequential()
    model.add(make_tensorflow_unet(3, 30))
    model.add(tf.keras.layers.Dense(30, activation=tf.nn.log_softmax))
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=config['adam_epsilon'])
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    # fit model on cityscapes data
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    (x_train, y_train), (x_test, y_test) = get_cityscapes()
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).with_options(options).batch(b)
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).with_options(options).batch(b)
    # train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False,
    #                         data_dir='/home/mzvyagin/datasets/')
    # train = train.with_options(options).batch(b)
    # test = test.with_options(options).batch(b)
    #res = model.fit(train, epochs=config['epochs'], batch_size=b)
    res = model.fit(train, epochs=config['epochs'])
    res_test = model.evaluate(test)
    return res_test[1], model


# same model just using gis data instead
def gis_tf_objective(config, classes=1):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.random.set_seed(0)
    keras.backend.set_image_data_format('channels_last')
    b = int(config['batch_size'])
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5",
                                                       "/gpu:6", "/gpu:7"])
    with strategy.scope():
    # model = keras.models.Sequential()
    # model.add(make_tensorflow_unet(4, 1))
    # model.add(keras.layers.Dense(1, activation="sigmoid"))
        model = sm.Unet('resnet34', activation="sigmoid")
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])
    # fit model on gis data
    (x_train, y_train), (x_test, y_test) = tf_gis_test_train_split()
    print(len(x_train))
    print(len(x_test))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).with_options(options).batch(b, drop_remainder=True)
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).with_options(options).batch(b, drop_remainder=True)
    res = model.fit(train, epochs=config['epochs'], batch_size=b)
    res_test = model.evaluate(test)
    return res_test[1], model


def get_cityscapes():
    """ Returns test, train split of Cityscapes data"""
    # first try loading from cache object, otherwise load from scratch

    # train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False,
    #                         data_dir='/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/')
    train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False,
                            data_dir='/home/mzvyagin/datasets/')
    train = list(train)
    train_x = [pair['image_left'] for pair in train]
    train_y = [pair['segmentation_label'] for pair in train]
    train_x = list(map(lambda x: x.numpy()/255.0, train_x))
    # train_x, train_y = [], []
    # for i in train:
    #     train_x.append(i['image_left'].numpy() / 255)
    #     train_y.append(i['segmentation_label'].numpy() / 255)
    #test_x, test_y = [], []
    test = list(test)
    test_x = [pair['image_left'] for pair in test]
    test_y = [pair['segmentation_label'] for pair in test]
    train_x = list(map(lambda x: tf.convert_to_tensor(x.numpy()/255.0), test_x))
    # for i in test:
    #     test_x.append(i['image_left'].numpy() / 255)
    #     test_y.append(i['segmentation_label'].numpy() / 255)
    return (train_x, train_y), (test_x, test_y)
    #return train, test


if __name__ == "__main__":
    test_config = {'batch_size': 1, 'learning_rate': .001, 'epochs': 1}
    #res = cityscapes_tf_objective(test_config)
    # print(res[0])
    res = gis_tf_objective(test_config)
