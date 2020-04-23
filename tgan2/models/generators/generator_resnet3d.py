import collections
import math

import chainer
import chainer.functions as F
import chainer.links as L
import numpy

from tgan2.utils import make_batch_normalization


class GeneratorResNet3D(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, out_channels=3):
        self.n_frames = n_frames
        self.z_slow_dim = z_slow_dim
        self.out_channels = out_channels
        super(GeneratorResNet3D, self).__init__()
        with self.init_scope():
            self.dc0 = L.Linear(z_slow_dim, 512)
            self.dc1 = GenBlock3D(512, 256, upsample=(1, 2, 2))
            self.dc2 = GenBlock3D(256, 128, upsample=(1, 2, 2))
            self.dc3 = GenBlock3D(128, 128, upsample=(2, 2, 2))
            self.dc4 = GenBlock3D(128, 64, upsample=(2, 2, 2))
            self.dc5 = GenBlock3D(64, 64, upsample=(2, 2, 2))
            self.dc6 = GenBlock3D(64, out_channels, upsample=(2, 2, 2))

    def make_hidden(self, batchsize):
        z_slow = chainer.Variable(self.xp.asarray(
            numpy.random.uniform(
                -1, 1, (batchsize, self.z_slow_dim)).astype(numpy.float32)))
        return z_slow

    def __call__(self, z_slow, n_frames=None):
        assert(n_frames is None or n_frames == 16)
        h = self.dc0(z_slow).reshape(len(z_slow), -1, 1, 1, 1)
        h = self.dc1(h)
        h = self.dc2(h)
        h = self.dc3(h)
        h = self.dc4(h)
        h = self.dc5(h)
        x = self.dc6(h)
        return x


class GenBlock3D(chainer.Chain):

    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation='relu', upsample=None, n_classes=0):
        super(GenBlock3D, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = getattr(F, activation)
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        with self.init_scope():
            self.c1 = L.ConvolutionND(
                3, in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.ConvolutionND(
                3, hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = L.ConvolutionND(
                    3, in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

            self.b1 = make_batch_normalization(in_channels, n_classes=n_classes)
            self.b2 = make_batch_normalization(hidden_channels, n_classes=n_classes)

    def residual(self, x, c=None):
        h = x
        h = self.b1(h, c) if c is not None else self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1, self.upsample) if self.upsample else self.c1(h)
        h = self.b2(h, c) if c is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc, self.upsample) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x, c=None):
        return self.residual(x, c) + self.shortcut(x)


def _upsample(x, upsample):
    if isinstance(upsample, collections.Iterable):
        outsize = tuple(d * s for d, s in zip(x.shape[2:], upsample))
    else:
        outsize = tuple(d * upsample for d in x.shape[2:])
    return F.unpooling_nd(x, upsample, outsize=outsize)


def upsample_conv(x, conv, upsample):
    return conv(_upsample(x, upsample))
