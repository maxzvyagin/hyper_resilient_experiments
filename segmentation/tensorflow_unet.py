### Tensorflow UNet with Resnet34 Backbone
import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds


def cityscapes_tf_objective(config, classes=20):
    model = sm.Unet('resnet34', encoder_weights=None, classes=classes)
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # fit model on cityscapes data
    (x_train, y_train), (x_test, y_test) = get_cityscapes()
    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    res_test = model.evaluate(x_test, y_test)
    return res_test[1], model


# implement this later
def gis_tf_objective(config, classes=1):
    pass


@tf.function
def get_cityscapes():
    """ Returns test, train split of Cityscapes data"""
    train, test = tfds.load('cityscapes', split=['train', 'test'], shuffle_files=False)
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
    with tf.device('/device:GPU:7'):
        test_config = {'batch_size': 1, 'learning_rate': .001, 'epochs': 1}
        res = cityscapes_tf_objective(test_config)
