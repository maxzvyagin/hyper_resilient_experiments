### Tensorflow UNet with Resnet34 Backbone
import segmentation_models as sm
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from gis_preprocess import tf_gis_test_train_split

def cityscapes_tf_objective(config, classes=20):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4"])
    with strategy.scope():
        model = sm.Unet('resnet34', encoder_weights=None, classes=classes)
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    # fit model on cityscapes data
    (x_train, y_train), (x_test, y_test) = tf_gis_test_train_split()
    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    res_test = model.evaluate(x_test, y_test)
    return res_test[1], model


# same model just using gis data instead
def gis_tf_objective(config, classes=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:6"])
    with strategy.scope():
        model = sm.Unet('resnet34', encoder_weights=None, classes=classes)
        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    # fit model on cityscapes data
    (x_train, y_train), (x_test, y_test) = get_cityscapes()
    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    res_test = model.evaluate(x_test, y_test)
    return res_test[1], model

def get_cityscapes():
    """ Returns test, train split of Cityscapes data"""
    train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False,
                            data_dir='/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/')
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
    test_config = {'batch_size': 1, 'learning_rate': .001, 'epochs': 1}
    #res = cityscapes_tf_objective(test_config)
    #print(res[0])
    res = gis_tf_objective(test_config)
