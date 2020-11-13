import tensorflow as tf
import os

def mnist_tf_objective(config):
    tf.random.set_seed(0)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    mnist = tf.keras.datasets.mnist
    b = int(config['batch_size'])
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(b)
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(b)
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # train = train.with_options(options).batch(b)
    # test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).with_options(options).batch(b)
    # # gpus = tf.config.experimental.list_physical_devices('GPU')
    # # tf.config.experimental.set_visible_devices(gpus[4:8], 'GPU')
    # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5",
    #                                                    "/gpu:6", "/gpu:7"])
    # with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(config['dropout']),
        # need to use log softmax since that's what pytorch uses in nn.CrossEntropyLoss()
        tf.keras.layers.Dense(10, activation=tf.nn.log_softmax)
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    res = model.fit(train, epochs=config['epochs'], batch_size=b)
    res_test = model.evaluate(test)
    return (res_test[1], model)

if __name__ == "__main__":
    test_config = {'batch_size': 1000, 'learning_rate': .001, 'epochs': 1, 'dropout': 0.5}
    res = mnist_tf_objective(test_config)