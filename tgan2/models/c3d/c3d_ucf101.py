from __future__ import print_function

import collections
import os

import chainer
from chainer.dataset import download
import chainer.functions as F
from chainer.functions.activation.relu import relu
from chainer.functions.activation.softmax import softmax
from chainer.functions.noise.dropout import dropout
from chainer.functions.pooling.max_pooling_nd import max_pooling_nd
from chainer.initializers import constant
from chainer.initializers import normal
from chainer.links import Bias
from chainer.links.connection.convolution_nd import ConvolutionND
from chainer.links.connection.linear import Linear
from chainer.serializers import npz
import numpy


class C3DVersion1UCF101(chainer.Chain):

    n_frames = 16
    rows = 64
    cols = 64

    def __init__(self, pretrained_model='auto', n_channels=3, n_outputs=101, mean_path='datasets/models/mean2.npz'):
        super(C3DVersion1UCF101, self).__init__()
        if pretrained_model:
            # As a sampling process is time-consuming,
            # we employ a zero initializer for faster computation.
            init = constant.Zero()
            conv_kwargs = {'initialW': init, 'initial_bias': init}
            fc_kwargs = conv_kwargs
        else:
            # employ default initializers used in the original paper
            conv_kwargs = {
                'initialW': normal.Normal(0.01),
                'initial_bias': constant.Zero(),
            }
            fc_kwargs = {
                'initialW': normal.Normal(0.005),
                'initial_bias': constant.One(),
            }
        with self.init_scope():
            self.conv1a = ConvolutionND(3, n_channels, 64, 3, 1, 1, **conv_kwargs)
            self.conv2a = ConvolutionND(3, 64, 128, 3, 1, 1, **conv_kwargs)
            self.conv3a = ConvolutionND(3, 128, 256, 3, 1, 1, **conv_kwargs)
            self.conv3b = ConvolutionND(3, 256, 256, 3, 1, 1, **conv_kwargs)
            self.conv4a = ConvolutionND(3, 256, 512, 3, 1, 1, **conv_kwargs)
            self.conv4b = ConvolutionND(3, 512, 512, 3, 1, 1, **conv_kwargs)
            self.conv5a = ConvolutionND(3, 512, 512, 3, 1, 1, **conv_kwargs)
            self.conv5b = ConvolutionND(3, 512, 512, 3, 1, 1, **conv_kwargs)
            self.fc6 = Linear(512 * 4 * 4, 4096, **fc_kwargs)
            self.fc7 = Linear(4096, 4096, **fc_kwargs)
            self.fc8 = Linear(4096, n_outputs, **fc_kwargs)
        if pretrained_model == 'auto':
            _retrieve(
                'conv3d_deepnetA_ucf.npz',
                'http://vlg.cs.dartmouth.edu/c3d/'
                'c3d_ucf101_finetune_whole_iter_20000',
                self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

        self.pre = ConvolutionND(3, n_channels, n_channels, 1, 1, 0, nobias=True, **conv_kwargs)
        self.pre.W.data[:] = 0
        self.pre.W.data[[0, 1, 2], [2, 1, 0]] = 128
        # self.pre.b.data[:] = 128 - numpy.array([90.25164795, 97.65701294, 101.4083252])
        self.mean = Bias(shape=(3, 16, 112, 112))
        mean = numpy.load(mean_path)['mean']
        self.mean.b.data[:] = 128 - mean[:, :, 8:8 + 112, 8:8 + 112]
        self.functions = collections.OrderedDict([
            ('pre', [self.pre, _resize, self.mean]),
            ('conv1a', [self.conv1a, relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2a', [self.conv2a, relu]),
            ('pool2', [_max_pooling_3d]),
            ('conv3a', [self.conv3a, relu]),
            ('conv3b', [self.conv3b, relu]),
            ('pool3', [_max_pooling_3d]),
            ('conv4a', [self.conv4a, relu]),
            ('conv4b', [self.conv4b, relu]),
            ('pool4', [_max_pooling_3d]),
            ('conv5a', [self.conv5a, relu]),
            ('conv5b', [self.conv5b, relu]),
            ('pool5', [_max_pooling_3d, dropout]),
            ('fc6', [self.fc6, relu, dropout]),
            ('fc7', [self.fc7, relu, dropout]),
            ('fc8', [self.fc8]),
            ('prob', [softmax]),
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.pre.to_gpu(device)
        self.mean.to_gpu(device)
        return self

    def to_cpu(self):
        super().to_cpu()
        self.pre.to_cpu()
        self.mean.to_cpu()
        return self

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.

        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As caffe_function uses shortcut symbols,
        # we import it here.
        from chainer.links.caffe import caffe_function
        caffe_pb = caffe_function.caffe_pb

        caffemodel = caffe_pb.NetParameter()
        with open(path_caffemodel, 'rb') as model_file:
            caffemodel.MergeFromString(model_file.read())
        chainermodel = cls(pretrained_model=None)
        _transfer(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def extract(self, x, layers=['prob']):
        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def __call__(self, x):
        return self.extract(x, layers=['fc8'])['fc8']

    def get_feature(self, x):
        return self.extract(x, layers=['fc7'])['fc7']


def _max_pooling_3d(x):
    # print(x.data.shape)
    return max_pooling_nd(x, ksize=2)
    # return max_pooling_nd(x, ksize=2, stride=2)


def _max_pooling_2d(x):
    return max_pooling_nd(x, ksize=(1, 2, 2))
    # return max_pooling_nd(x, ksize=(1, 2, 2), stride=(1, 2, 2))


def _resize(x):
    # N, C, T, H, W
    N, C, T, H, W = x.shape
    x = x.transpose([0, 2, 1, 3, 4]).reshape(N * T, C, H, W)
    x = F.resize_images(x, (112, 112))
    x = x.reshape(N, T, C, 112, 112).transpose(0, 2, 1, 3, 4)
    return x


def _transfer(caffemodel, chainermodel):

    def transfer_layer(src, dst):
        dst.W.data.ravel()[:] = src.blobs[0].diff
        dst.b.data.ravel()[:] = src.blobs[1].diff

    layers = {l.name: l for l in caffemodel.layers}
    print([l.name for l in caffemodel.layers])
    transfer_layer(layers['conv1a'], chainermodel.conv1a)
    transfer_layer(layers['conv2a'], chainermodel.conv2a)
    transfer_layer(layers['conv3a'], chainermodel.conv3a)
    transfer_layer(layers['conv3b'], chainermodel.conv3b)
    transfer_layer(layers['conv4a'], chainermodel.conv4a)
    transfer_layer(layers['conv4b'], chainermodel.conv4b)
    transfer_layer(layers['conv5a'], chainermodel.conv5a)
    transfer_layer(layers['conv5b'], chainermodel.conv5b)
    transfer_layer(layers['fc6'], chainermodel.fc6)
    transfer_layer(layers['fc7'], chainermodel.fc7)
    transfer_layer(layers['fc8'], chainermodel.fc8)

    # RGB-to-BGR -> scale -> subtract mean
    # chainermodel.pre.W.data[:] = 0
    # chainermodel.pre.W.data[[0, 1, 2], [2, 1, 0]] = 128
    # chainermodel.pre.b.data[:] = 128 - numpy.array([96.35598485, 104.3764703, 109.10312843])


def _make_npz(path_npz, url, model):
    path_caffemodel = "/mnt/sakura201/mattya/c3d/c3d_ucf101_finetune_whole_iter_20000"
    # path_caffemodel = "c3d_ucf101_finetune_whole_iter_20000"
    print('Now loading caffemodel (usually it may take few minutes)')
    C3DVersion1UCF101.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))
