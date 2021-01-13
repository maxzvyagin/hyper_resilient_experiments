import tensorflow as tf
from hyper_resilient_experiments.segmentation.gis_preprocess import (tf_gis_test_train_split, pt_gis_train_test_split,
                                                                     perturbed_tf_gis_test_data,
                                                                     perturbed_pt_gis_test_data)
from hyper_resilient_experiments.segmentation import pytorch_unet, tensorflow_unet


if __name__ == "__main__":
    test_config = {'batch_size': 50, 'learning_rate': .001, 'epochs': 1, 'adam_epsilon': 10 ** -9}
    # res = cityscapes_tf_objective(test_config)
    # print(res[0])
    # cityscapes_tf_objective(test_config)
    res = tensorflow_unet.gis_tf_objective(test_config)
    x_test, y_test = perturbed_tf_gis_test_data()
    test_acc = res[0].evaluate(x_test, y_test, batch_size=test_config['batch_size'])
    print(res[0])
    print(test_acc)