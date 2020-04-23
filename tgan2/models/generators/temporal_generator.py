import chainer
import chainer.functions as F
import chainer.links as L

from tgan2.utils import make_batch_normalization


class TemporalGenerator(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, z_fast_dim=256, n_classes=0):
        super(TemporalGenerator, self).__init__()
        w = None
        assert(n_frames == 16)
        assert(n_classes == 0)
        self.n_frames = n_frames
        with self.init_scope():
            self.dc0 = L.DeconvolutionND(1, z_slow_dim, 512, 1, 1, 0, initialW=w)
            self.dc1 = L.DeconvolutionND(1, 512, 256, 4, 2, 1, initialW=w)
            self.dc2 = L.DeconvolutionND(1, 256, 128, 4, 2, 1, initialW=w)
            self.dc3 = L.DeconvolutionND(1, 128, 128, 4, 2, 1, initialW=w)
            self.dc4 = L.DeconvolutionND(1, 128, z_fast_dim, 4, 2, 1, initialW=w)

            self.bn0 = make_batch_normalization(512, n_classes=0)
            self.bn1 = make_batch_normalization(256, n_classes=0)
            self.bn2 = make_batch_normalization(128, n_classes=0)
            self.bn3 = make_batch_normalization(128, n_classes=0)

        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

    def make_hidden(self, batchsize):
        xp = self.xp
        z_slow = chainer.Variable(
            xp.random.uniform(-1, 1, (batchsize, self.z_slow_dim)).astype(xp.float32))
        return z_slow

    def __call__(self, z_slow, c=None, n_frames=None):
        if (n_frames is not None) and (n_frames != self.n_frames):
            raise NotImplementedError
        if c is not None:
            raise NotImplementedError
        h = F.reshape(z_slow, (z_slow.shape[0], -1, 1))
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc4(h)).transpose(0, 2, 1)  # (N, n_frames, z_fast_dim)
        return z_fast
