import tensorflow as tf
from hyper_resilient_experiments.segmentation.gis_preprocess import tf_gis_test_train_split
from hyper_resilient_experiments.segmentation.UNet import make_tensorflow_unet

config = {'batch_size': 1, 'learning_rate': .001, 'epochs': 1, 'adam_epsilon': 10**-9}

files = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
                  "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
                 ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
                  "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
(x_train, y_train), (x_test, y_test) = tf_gis_test_train_split(img_and_shps=files)
train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)
model = tf.keras.Sequential()
model.add(make_tensorflow_unet(4, 1))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.log_softmax))
opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=config['adam_epsilon'])
model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.train(train, epochs=config['epochs'], batch_size=4)