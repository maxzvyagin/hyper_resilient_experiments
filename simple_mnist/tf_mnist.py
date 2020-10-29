import tensorflow as tf

def mnist_tf_objective(config):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])
    with strategy.scope():
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

    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    res_test = model.evaluate(x_test, y_test)
    return (res_test[1], model)