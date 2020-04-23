import chainer
import chainer.functions as F
import chainer.links as L

import numpy


class LSTMTemporalGenerator(chainer.Chain):

    """Frame seed generator using LSTM blocks.

    Args:
        n_frames (int): The num of generated seeds.
        z_slow_dim (int): The num of units of latent variables in all LSTM blocks.
        z_fast_dim (int): The dim of output array.
        n_layers (int): How many LSTMs are stacked.
        dropout (float): The ratio of dropout which is appied to the inputs of LSTMs.

    """

    def __init__(self, n_frames=16, z_slow_dim=256, z_fast_dim=256, n_layers=1,
                 dropout=0.0, stride=1, n_classes=0):
        super(LSTMTemporalGenerator, self).__init__()
        self.n_frames = n_frames
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.stride = stride
        self.decoders = []
        self.n_classes = n_classes
        with self.init_scope():
            for i in range(self.n_layers):
                dim = z_slow_dim if 0 < i else (z_slow_dim + self.n_classes)
                decoder = L.LSTM(dim, z_slow_dim)
                setattr(self, 'decoder{}'.format(i), decoder)
                self.decoders.append(decoder)
            if z_slow_dim != z_fast_dim:
                self.linear = L.Linear(z_slow_dim, z_fast_dim)

    def make_hidden(self, batchsize):
        z_slow = chainer.Variable(self.xp.asarray(
            numpy.random.uniform(
                -1, 1, (batchsize, self.z_slow_dim)).astype(numpy.float32)))
        return z_slow

    def make_in_val(self, i, z_slow, c=None):
        if self.n_classes == 0:
            assert(c is None)
            if i == 0:
                return z_slow
            else:
                return chainer.Variable(self.xp.zeros_like(z_slow.array))
        else:
            assert(c is not None)
            if i == 0:
                c_array = chainer.Variable(self.xp.eye(
                    self.n_classes, dtype=self.xp.float32)[c.array])
                return F.concat([z_slow, c_array])
            else:
                return chainer.Variable(self.xp.zeros(
                    (len(z_slow), self.z_slow_dim + self.n_classes),
                    dtype=self.xp.float32))

    def __call__(self, z_slow, c=None, n_frames=None):
        """Generate frame seed from latent variable z_slow.

        Args:
            z_slow (ndarray): The shape is (N, z_slow_dim).

        Returns:
            z_fast (ndarray): The shape is (N, n_frames, z_fast_dim).

        """
        n_frames = n_frames if n_frames is not None else self.n_frames
        # Use stride=1 in test mode
        stride = self.stride if chainer.config.train else 1

        for decoder in self.decoders:
            decoder.reset_state()

        ys = []
        total_frames = (n_frames - 1) * stride + 1
        for i in range(total_frames):
            h = self.make_in_val(i, z_slow, c)
            for decoder in self.decoders:
                h = decoder(F.dropout(h, ratio=self.dropout))

            if i % stride == 0:
                y = h if self.z_slow_dim == self.z_fast_dim else self.linear(h)
                ys.append(y)

        ys = F.concat(ys)
        ys = ys.reshape(len(ys), n_frames, self.z_fast_dim)
        if self.z_slow_dim == self.z_fast_dim:
            return ys
        else:
            return F.tanh(ys)
