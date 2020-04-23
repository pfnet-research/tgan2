import math

import chainer
import chainer.functions as F
import chainer.links as L


class ResNetFrameDiscriminator(chainer.Chain):

    def __init__(self, in_channels, mid_ch=64, n_classes=0, activation='relu'):
        super(ResNetFrameDiscriminator, self).__init__()
        self.activation = getattr(F, activation)
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedDisBlock(in_channels, mid_ch)

            kwargs = {'activation': activation, 'downsample': True}
            self.block2 = DisBlock(mid_ch, mid_ch * 2, **kwargs)
            self.block3 = DisBlock(mid_ch * 2, mid_ch * 4, **kwargs)
            self.block4 = DisBlock(mid_ch * 4, mid_ch * 8, **kwargs)
            self.block5 = DisBlock(mid_ch * 8, mid_ch * 16, **kwargs)
            self.l6 = L.Linear(mid_ch * 16, 1, initialW=initializer)
            if n_classes > 0:
                self.l_c = L.EmbedID(n_classes, mid_ch * 16, initialW=initializer)

    def extract_feature(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        return h

    def __call__(self, x, c=None):
        h = self.extract_feature(x)
        output = self.l6(h)
        if c is not None:
            w_c = self.l_c(c)
            output += F.sum(w_c * h, axis=1, keepdims=True)
        return output


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
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

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


def _downsample(x):
    ksize = [(2 if 1 < k else 1) for k in x.shape[2:]]
    pad = [(0 if k % 2 == 0 else 1) for k in x.shape[2:]]
    return F.average_pooling_2d(x, tuple(ksize), pad=tuple(pad))


class OptimizedDisBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedDisBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.average_pooling_2d(h, (2, 2))
        return h

    def shortcut(self, x):
        h = F.average_pooling_2d(x, (2, 2))
        return self.c_sc(h)

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)
