import chainer
import chainer.functions as F
import chainer.links as L

from tgan2.models.resblocks import DisBlock
from tgan2.models.resblocks import OptimizedDisBlock


class ResNetVideoDiscriminator(chainer.Chain):

    def __init__(self, in_channels, mid_ch=64, n_classes=0, activation='relu'):
        super(ResNetVideoDiscriminator, self).__init__()
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
        h = F.sum(h, axis=(2, 3, 4))  # Global pooling
        return h

    def __call__(self, x, c=None):
        h = self.extract_feature(x)
        output = self.l6(h)
        if c is not None:
            w_c = self.l_c(c)
            output += F.sum(w_c * h, axis=1, keepdims=True)
        return output
