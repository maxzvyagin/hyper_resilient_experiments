### From the mxnet mnist tutorial
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import optimizer
from mxnet import autograd as ag

def mnist_mx_objective(config):
    # Fixing the random seed
    # mx.random.seed(42)
    mnist = mx.test_utils.get_mnist()
    train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], config['batch_size'])
    val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], config['batch_size'])

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))

    gpus = mx.test_utils.list_gpus()
    ctx = [mx.gpu(0)] if gpus else [mx.cpu(0)]
    net.initialize(mx.init.Uniform(scale=1), ctx=ctx)
    optim = optimizer.Adam(learning_rate=config['learning_rate'])
    trainer = gluon.Trainer(net.collect_params(), optim)

    epoch = config['epochs']
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(epoch):
    # Reset the train data iterator.
        train_data.reset()
        # Loop over the train data iterator.
        for batch in train_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        # Inside training scope
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                # Computes softmax cross entropy loss.
                loss = softmax_cross_entropy_loss(z, y)
                # Backpropagate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])
        # Gets the evaluation result.
        name, acc = metric.get()
        # Reset evaluation result to initial state.
        metric.reset()
        print('training acc at epoch %d: %s=%f' % (i, name, acc))

    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    # Reset the validation data iterator.
    val_data.reset()
    # Loop over the validation data iterator.
    for batch in val_data:
    # Splits validation data into multiple slices along batch_axis
    # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    # Splits validation label into multiple slices along batch_axis
    # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        # Updates internal evaluation
        metric.update(label, outputs)
    return (metric.get(), net)
    #print('validation acc: %s=%f' % metric.get())