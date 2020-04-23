import math
import warnings

import chainer
import chainer.functions as F
import chainer.links as L

from tgan2.models.bn.categorical_conditional_batch_normalization \
    import CategoricalConditionalBatchNormalization

try:
    from chainermn.links import MultiNodeBatchNormalization
except Exception:
    warnings.warn('To perform batch normalization with multiple GPUs or '
                  'multiple nodes, MultiNodeBatchNormalization link is '
                  'needed. Please install ChainerMN: '
                  'pip install chainermn')


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class GenBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation='relu', upsample=False, n_classes=0):
        super(GenBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = getattr(F, activation)
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        with self.init_scope():
            self.c1 = L.Convolution2D(
                in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(
                hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if n_classes > 0:
                if (not hasattr(chainer.config, 'comm')) or chainer.config.comm is None:
                    kwargs = {'n_cat': n_classes}
                else:
                    kwargs = {'n_cat': n_classes, 'comm': chainer.config.comm}
                self.b1 = CategoricalConditionalBatchNormalization(in_channels, **kwargs)
                self.b2 = CategoricalConditionalBatchNormalization(hidden_channels, **kwargs)
            else:
                if (not hasattr(chainer.config, 'comm')) or chainer.config.comm is None:
                    self.b1 = L.BatchNormalization(in_channels)
                    self.b2 = L.BatchNormalization(hidden_channels)
                else:
                    self.b1 = MultiNodeBatchNormalization(in_channels, comm=chainer.config.comm)
                    self.b2 = MultiNodeBatchNormalization(hidden_channels, comm=chainer.config.comm)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x, c=None):
        h = x
        h = self.b1(h, c) if c is not None else self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, c) if c is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def __call__(self, x, c=None):
        return self.residual(x, c) + self.shortcut(x)


def _downsample(x):
    ksize = [(2 if 1 < k else 1) for k in x.shape[2:]]
    pad = [(0 if k % 2 == 0 else 1) for k in x.shape[2:]]
    return F.average_pooling_nd(x, tuple(ksize), pad=tuple(pad))


class DisBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation='relu', downsample=False):
        super(DisBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = getattr(F, activation)
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = L.ConvolutionND(3, in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.ConvolutionND(3, hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = L.ConvolutionND(3, in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedDisBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedDisBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c1 = L.ConvolutionND(3, in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.ConvolutionND(3, out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = L.ConvolutionND(3, in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        # h = _downsample(h)
        h = F.average_pooling_nd(h, (1, 2, 2))
        return h

    def shortcut(self, x):
        # return self.c_sc(_downsample(x))
        h = F.average_pooling_nd(x, (1, 2, 2))
        return self.c_sc(h)

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)
