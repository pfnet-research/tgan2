import chainer
import chainer.functions as F
import chainer.links as L
import numpy

from tgan2.utils import make_batch_normalization


class GeneratorConv3D(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, out_channels=3):
        self.n_frames = n_frames
        self.z_slow_dim = z_slow_dim
        self.out_channels = out_channels
        super(GeneratorConv3D, self).__init__()
        w = None
        with self.init_scope():
            self.dc0 = L.DeconvolutionND(3, z_slow_dim, 512, 1, 1, 0, initialW=w)
            self.dc1 = L.DeconvolutionND(3, 512, 256, (1, 4, 4), (1, 2, 2), (0, 1, 1), initialW=w)
            self.dc2 = L.DeconvolutionND(3, 256, 128, (1, 4, 4), (1, 2, 2), (0, 1, 1), initialW=w)
            self.dc3 = L.DeconvolutionND(3, 128, 128, 4, 2, 1, initialW=w)
            self.dc4 = L.DeconvolutionND(3, 128, 64, 4, 2, 1, initialW=w)
            self.dc5 = L.DeconvolutionND(3, 64, 64, 4, 2, 1, initialW=w)
            self.dc6 = L.DeconvolutionND(3, 64, out_channels, 4, 2, 1, initialW=w)

            self.bn0 = make_batch_normalization(512)
            self.bn1 = make_batch_normalization(256)
            self.bn2 = make_batch_normalization(128)
            self.bn3 = make_batch_normalization(128)
            self.bn4 = make_batch_normalization(64)
            self.bn5 = make_batch_normalization(64)

    def make_hidden(self, batchsize):
        z_slow = chainer.Variable(self.xp.asarray(
            numpy.random.uniform(
                -1, 1, (batchsize, self.z_slow_dim)).astype(numpy.float32)))
        return z_slow

    def __call__(self, z_slow, n_frames=None):
        assert(n_frames is None or n_frames == 16)
        h = F.reshape(z_slow, (z_slow.shape[0], -1, 1, 1, 1))
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        h = F.relu(self.bn5(self.dc5(h)))
        z_fast = F.tanh(self.dc6(h))
        return z_fast
