import chainer
import chainer.functions as F
import chainer.links as L

from tgan2.models.conv_lstm import ConvLSTM
from tgan2.models.resblocks import GenBlock
from tgan2.utils import make_batch_normalization

import numpy


def _expand(option):
    if isinstance(option, (list, tuple)):
        return option
    elif option:
        return [2, 2, 2]
    else:
        return [1, 1, 1]


class MultiTGANCLSTMGenerator(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, out_channels=3, t_size=4, ch=32, clstm_ch=512,
                 n_layers=1, stride=1, max_offset=0, activation='relu', n_classes=0,
                 subsample_frame=True, subsample_batch=True, render_level=4,
                 use_avg_pooling=False, use_single_video=False):
        super(MultiTGANCLSTMGenerator, self).__init__()
        self.n_frames = n_frames
        self.z_slow_dim = z_slow_dim
        self.out_channels = out_channels
        self.t_size = t_size
        self.clstm_ch = clstm_ch
        self.ch = ch
        self.n_classes = n_classes
        self.use_avg_pooling = use_avg_pooling
        self.use_single_video = use_single_video

        self.subsample_frame = _expand(subsample_frame)
        self.subsample_batch = _expand(subsample_batch)
        self.render_level = render_level

        self.decoders = []
        self.n_layers = n_layers
        self.stride = stride
        self.max_offset = max_offset
        self.activation = getattr(F, activation)
        with self.init_scope():
            self.fc = L.Linear(z_slow_dim, t_size * t_size * clstm_ch)
            for i in range(self.n_layers):
                decoder = ConvLSTM(clstm_ch, clstm_ch, 3, 1, 1, peephole=False)
                setattr(self, 'decoder{}'.format(i), decoder)
                self.decoders.append(decoder)
            kwargs = {'activation': activation, 'upsample': True, 'n_classes': n_classes}

            # Level 1
            self.block1 = GenBlock(clstm_ch, ch * 16, **kwargs)
            self.block2 = GenBlock(ch * 16, ch * 8, **kwargs)
            self.block3 = GenBlock(ch * 8, ch * 4, **kwargs)
            self.b3 = make_batch_normalization(ch * 4, n_classes=n_classes)
            self.c3 = L.Convolution2D(ch * 4, out_channels, 3, 1, 1)

            # Level 2
            self.block4 = GenBlock(ch * 4, ch * 2, **kwargs)
            self.b4 = make_batch_normalization(ch * 2, n_classes=n_classes)
            self.c4 = L.Convolution2D(ch * 2, out_channels, 3, 1, 1)

            # Level 3
            self.block5 = GenBlock(ch * 2, ch, **kwargs)
            self.b5 = make_batch_normalization(ch, n_classes=n_classes)
            self.c5 = L.Convolution2D(ch, out_channels, 3, 1, 1)

            # Level 4
            self.block6 = GenBlock(ch, ch, **kwargs)
            self.b6 = make_batch_normalization(ch, n_classes=n_classes)
            self.c6 = L.Convolution2D(ch, out_channels, 3, 1, 1)

    def make_hidden(self, batchsize):
        z_slow = chainer.Variable(self.xp.asarray(
            numpy.random.uniform(
                -1, 1, (batchsize, self.z_slow_dim)).astype(numpy.float32)))
        return z_slow

    def make_in_val(self, t, z_slow):
        N = z_slow.shape[0]
        if t == 0:
            return self.fc(z_slow).reshape(N, self.clstm_ch, self.t_size, self.t_size)
        else:
            return chainer.Variable(
                self.xp.zeros((N, self.clstm_ch, self.t_size, self.t_size), dtype=self.xp.float32))

    def separate(self, h, N):
        NT, C, H, W = h.shape
        T = NT // N
        return h.reshape(N, T, C, H, W)

    def join(self, h):
        N, T, C, H, W = h.shape
        return h.reshape(N * T, C, H, W), N

    def subsample(self, h, batch=2, frame=2):
        if self.use_avg_pooling:
            N, T, C, H, W = h.shape
            Td = T // frame
            h = h.reshape(N, Td, frame, C, H, W)
            h = F.mean(h, axis=2)
            return h[::batch]
        else:
            return h[::batch, numpy.random.randint(frame)::frame]

    def render(self, link_bn, link_conv, h, N):
        NT, C, H, W = h.shape
        T = NT // N
        h = self.activation(link_bn(h))
        h = link_conv(h)
        H, W = h.shape[2:]
        h = F.tanh(h).reshape(N, T, self.out_channels, H, W)
        h = h.transpose(0, 2, 1, 3, 4)
        return h

    def __call__(self, z_slow, n_frames=None):
        n_frames = n_frames if n_frames is not None else self.n_frames
        # Use stride=1 in test mode
        stride = self.stride if chainer.config.train else 1
        offset = numpy.random.randint(self.max_offset + 1) if chainer.config.train else 0

        for decoder in self.decoders:
            decoder.reset_state()

        hs = []
        total_frames = offset + (n_frames - 1) * stride + 1
        for t in range(total_frames):
            h = self.make_in_val(t, z_slow)
            for decoder in self.decoders:
                h = decoder(h)
            td = t - offset
            if 0 <= td and td % stride == 0:
                hs.append(h)
        N, C, H, W = hs[0].shape
        h = F.concat(hs)  # (N, T * C, H, W)
        h = h.reshape(N * n_frames, C, H, W)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        if chainer.config.train:
            x1 = self.render(self.b3, self.c3, h, N)
            h, N = self.join(self.subsample(
                self.separate(h, N),
                frame=self.subsample_frame[0], batch=self.subsample_batch[0]))
            h = self.block4(h)
            x2 = self.render(self.b4, self.c4, h, N)
            h, N = self.join(self.subsample(
                self.separate(h, N),
                frame=self.subsample_frame[1], batch=self.subsample_batch[1]))
            h = self.block5(h)
            x3 = self.render(self.b5, self.c5, h, N)
            h, N = self.join(self.subsample(
                self.separate(h, N),
                frame=self.subsample_frame[2], batch=self.subsample_batch[2]))
            h = self.block6(h)
            x4 = self.render(self.b6, self.c6, h, N)
            if self.use_single_video:
                return x4
            else:
                return (x1, x2, x3, x4)
        else:
            assert(1 <= self.render_level <= 4)
            for i in range(self.render_level - 1):
                block = getattr(self, 'block{}'.format(4 + i))
                h = block(h)
            bn = getattr(self, 'b{}'.format(2 + self.render_level))
            conv = getattr(self, 'c{}'.format(2 + self.render_level))
            return self.render(bn, conv, h, N)
