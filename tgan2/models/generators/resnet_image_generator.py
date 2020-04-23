import chainer
import chainer.functions as F
import chainer.links as L

from tgan2.models.resblocks import GenBlock
from tgan2.utils import make_batch_normalization


class ResNetImageGenerator(chainer.Chain):

    def __init__(self, z_slow_dim, z_fast_dim, out_channels, bottom_width,
                 ch=64, activation='relu', n_classes=0):
        super(ResNetImageGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim
        self.out_channels = out_channels
        self.bottom_width = bottom_width
        self.ch = ch
        self.activation = getattr(F, activation)
        self.n_classes = n_classes
        slow_mid_dim = bottom_width * bottom_width * ch * 8
        fast_mid_dim = bottom_width * bottom_width * ch * 8
        with self.init_scope():
            self.l0s = L.Linear(z_slow_dim, slow_mid_dim, initialW=initializer)
            self.l0f = L.Linear(z_fast_dim, fast_mid_dim, initialW=initializer)

            kwargs = {'activation': activation, 'upsample': True, 'n_classes': n_classes}
            self.block2 = GenBlock(ch * 16, ch * 8, **kwargs)
            self.block3 = GenBlock(ch * 8, ch * 4, **kwargs)
            self.block4 = GenBlock(ch * 4, ch * 2, **kwargs)
            self.block5 = GenBlock(ch * 2, ch, **kwargs)
            self.b6 = make_batch_normalization(ch, n_classes=n_classes)
            self.l6 = L.Convolution2D(
                ch, out_channels, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, z_slow, z_fast, c=None):
        if self.n_classes == 0:
            assert(c is None)
        else:
            assert(c is not None)
        n = z_slow.shape[0]
        h_slow = F.reshape(self.l0s(z_slow), (n, self.ch * 8, self.bottom_width, self.bottom_width))
        h_fast = F.reshape(self.l0f(z_fast), (n, self.ch * 8, self.bottom_width, self.bottom_width))
        h = F.concat([h_slow, h_fast], axis=1)
        h = self.block2(h, c)
        h = self.block3(h, c)
        h = self.block4(h, c)
        h = self.block5(h, c)
        h = self.b6(h, c) if c is not None else self.b6(h)
        h = self.activation(h)
        h = F.tanh(self.l6(h))
        return h
