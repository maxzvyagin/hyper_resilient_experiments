### Tensorflow UNet with Resnet34 Backbone
import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

def cityscapes_tf_objective(config):
    model = sm.Unet()
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # fit model on cityscapes data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cityscapes.load_data()
    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    res_test = model.evaluate(x_test, y_test)
    return (res_test[1], model)

# implement this later
def gis_tf_objective(config):
    pass
