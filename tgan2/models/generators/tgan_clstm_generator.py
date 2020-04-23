import chainer
import chainer.functions as F
import chainer.links as L

from tgan2.models.conv_lstm import ConvLSTM
from tgan2.models.resblocks import GenBlock
from tgan2.utils import make_batch_normalization


class TGANCLSTMGenerator(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, out_channels=3,
                 n_layers=1, stride=1, activation='relu', n_classes=0):
        super(TGANCLSTMGenerator, self).__init__()
        self.n_frames = n_frames
        self.z_slow_dim = z_slow_dim
        self.out_channels = out_channels
        self.n_classes = n_classes

        self.decoders = []
        self.n_layers = n_layers
        self.stride = stride
        self.activation = getattr(F, activation)
        with self.init_scope():
            self.fc = L.Linear(z_slow_dim, 4 * 4 * 512)
            for i in range(self.n_layers):
                decoder = ConvLSTM(512, 512, 3, 1, 1, peephole=False)
                setattr(self, 'decoder{}'.format(i), decoder)
                self.decoders.append(decoder)
            kwargs = {'activation': activation, 'upsample': True, 'n_classes': n_classes}
            self.block1 = GenBlock(512, 512, **kwargs)
            self.block2 = GenBlock(512, 256, **kwargs)
            self.block3 = GenBlock(256, 128, **kwargs)
            self.block4 = GenBlock(128, 64, **kwargs)
            self.b5 = make_batch_normalization(64, n_classes=n_classes)
            self.c5 = L.Convolution2D(64, out_channels, 3, 1, 1)

    def make_hidden(self, batchsize):
        xp = self.xp
        z_slow = chainer.Variable(
            xp.random.uniform(-1, 1, (batchsize, self.z_slow_dim)).astype(xp.float32))
        return z_slow

    def make_in_val(self, t, z_slow):
        N = z_slow.shape[0]
        if t == 0:
            return self.fc(z_slow).reshape(N, 512, 4, 4)
        else:
            return chainer.Variable(
                self.xp.zeros((N, 512, 4, 4), dtype=self.xp.float32))

    def __call__(self, z_slow, n_frames=None):
        n_frames = n_frames if n_frames is not None else self.n_frames
        # Use stride=1 in test mode
        stride = self.stride if chainer.config.train else 1

        for decoder in self.decoders:
            decoder.reset_state()

        N = z_slow.shape[0]
        hs = []
        total_frames = (n_frames - 1) * stride + 1
        for t in range(total_frames):
            h = self.make_in_val(t, z_slow)
            for decoder in self.decoders:
                h = h + decoder(h)
            if t % stride == 0:
                hs.append(h)
        N, C, H, W = hs[0].shape
        h = F.concat(hs)  # (N, T * C, H, W)
        h = h.reshape(N * n_frames, C, H, W)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)

        h = self.activation(self.b5(h))
        h = self.c5(h)
        H, W = h.shape[2:]
        h = F.tanh(h).reshape(N, n_frames, self.out_channels, H, W)
        h = h.transpose(0, 2, 1, 3, 4)
        return h
