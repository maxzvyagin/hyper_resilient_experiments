import tensorflow as tf
from hyper_resilient_experiments.segmentation.gis_preprocess import tf_gis_test_train_split

files = [("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2012_refl.img",
                  "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2012_merge.shp"),
                 ("/scratch/mzvyagin/Ephemeral_Channels/Imagery/vhr_2014_refl.img",
                  "/scratch/mzvyagin/Ephemeral_Channels/Reference/reference_2014_merge.shp")]
(x_train, y_train), (x_test, y_test) = tf_gis_test_train_split(img_and_shps=files)
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
