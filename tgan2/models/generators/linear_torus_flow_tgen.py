import chainer
import chainer.functions as F
import numpy


class LinearTorusFlowTemporalGenerator(chainer.Chain):

    def __init__(self, n_frames=16, z_slow_dim=256, z_fast_dim=256, n_classes=0):
        super(LinearTorusFlowTemporalGenerator, self).__init__()
        with self.init_scope():
            self.velocity = chainer.Parameter(
                numpy.zeros(z_fast_dim, dtype=numpy.float32))

        self.n_frames = n_frames
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

    def make_hidden(self, batchsize, Nz=None):
        xp = self.xp
        Nz = self.z_slow_dim if Nz is None else Nz
        z_slow = chainer.Variable(
            xp.random.uniform(0, 1, (batchsize, Nz)).astype(xp.float32))
        return z_slow

    def __call__(self, z_slow, c=None, n_frames=None):
        if c is not None:
            raise NotImplementedError

        # Do not use z_slow instead we radomly sample z_fast
        xp = self.xp
        T = n_frames if n_frames is not None else self.n_frames
        N = len(z_slow)
        Nz = self.z_fast_dim

        z_fast = self.make_hidden(N, Nz=Nz)
        initial = F.broadcast_to(z_fast.reshape(N, Nz, 1), (N, Nz, T))
        step = chainer.Variable(xp.arange(0, T, dtype=xp.float32))
        step = F.broadcast_to(step.reshape(1, 1, T), (N, Nz, T))
        v = F.broadcast_to(self.velocity.reshape(1, Nz, 1), (N, Nz, T))

        z_fast = initial + step * v
        ones = chainer.Variable(xp.ones_like(z_fast.array, dtype=xp.float32))
        z_fast = F.fmod(z_fast, ones)  # (N, z_fast_dim, n_frames)

        theta = (2 * numpy.pi) * z_fast
        z_fast = F.concat([F.cos(theta), F.sin(theta)])
        return z_fast.transpose(0, 2, 1)  # (N, n_frames, 2 * z_fast_dim)
