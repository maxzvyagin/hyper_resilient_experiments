### Definition of CIFAR100 Alexnet in MXNet
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import optimizer
from mxnet import autograd as ag
from mxnet.gluon.data import vision
from tqdm import tqdm

class MXNet_AlexNet(nn.Block):
    r"""AlexNet model from the `"One weird trick..." `_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, config, classes=1000, **kwargs):
        super(MXNet_AlexNet, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(nn.Conv2D(64, kernel_size=11, strides=4,
                               padding=5, activation='relu', in_channels=3, layout="NHWC"))
        self.blk.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        self.blk.add(nn.Conv2D(256, kernel_size=5, padding=2,
                               activation='relu', layout="NHWC"))
        self.blk.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        self.blk.add(nn.Conv2D(384, kernel_size=3, padding=1,
                               activation='relu', layout="NHWC"))
        self.blk.add(nn.Conv2D(256, kernel_size=3, padding=1,
                               activation='relu', layout="NHWC"))
        self.blk.add(nn.Conv2D(256, kernel_size=3, padding=1,
                               activation='relu', layout="NHWC"))
        self.blk.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        self.blk.add(nn.Flatten())
        self.blk.add(nn.Dense(4096, activation='relu'))
        self.blk.add(nn.Dropout(config['dropout']))
        self.blk.add(nn.Dense(4096, activation='relu'))
        self.blk.add(nn.Dropout(config['dropout']))
        self.blk.add(nn.Dense(classes))

    def forward(self, x):
        return self.blk(x)

def test(ctx, val_data, net):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])
        output = net(data)
        try:
            metric.update(label, output)
        ### Need to figure out why on earth this happens and why this except block is needed, I'm at a loss ###
        except:
            print("Exception")
            metric.update(label, output)
    return metric.get()

def transform(data, label):
    data = data.astype('float32')/255
    return data, label

def cifar_mxnet_objective(config):
    #net = MXNet_AlexNet(config)
    net = gluoncv.model_zoo.get_model('alexnet', classes=1000, pretrained=False)
    gpus = mx.test_utils.list_gpus()
    ctx = [mx.gpu(0)] if gpus else [mx.cpu(0)]
    net.initialize(mx.init.Uniform(scale=1), ctx=ctx)
    optim = optimizer.Adam(learning_rate=config['learning_rate'])
    trainer = gluon.Trainer(net.collect_params(), optim)

    train_data = gluon.data.DataLoader(vision.datasets.CIFAR100(train=True, transform=transform),
                                       batch_size=config['batch_size'], shuffle=False)
    val_data = gluon.data.DataLoader(vision.datasets.CIFAR100(train=False, transform=transform),
                                     batch_size=config['batch_size'], shuffle=False)

    #     # Use Accuracy as the evaluation metric.
    #     metric = mx.metric.Accuracy()
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in tqdm(range(config['epochs'])):
        for data, label in train_data:
            # forward + backward
            with ag.record():
                output = net(data)
                loss = criterion(output, label)
            loss.backward()
            # update parameters
            trainer.step(config['batch_size'])

    # Evaluate on Validation data
    name, val_acc = test(ctx, val_data, net)
    return val_acc, net


if __name__ == "__main__":
    config = {'learning_rate':.001, 'dropout':0.5, 'batch_size':100, 'epochs':5}
    acc, net = cifar_mxnet_objective(config)
    print("Final Model Accuracy: ", acc)