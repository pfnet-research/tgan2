import chainer
import chainer.functions as F

from tgan2.utils import make_instance


class TGANGenerator(chainer.Chain):

    def __init__(self, tgen, igen, out_channels=3, n_classes=0):
        super(TGANGenerator, self).__init__()
        self.n_classes = n_classes
        if isinstance(tgen, dict):
            import tgan2
            tgen = make_instance(tgan2, tgen, args={'n_classes': n_classes})
            igen = make_instance(
                tgan2, igen,
                args={'n_classes': n_classes, 'out_channels': out_channels})

        with self.init_scope():
            self.tgen = tgen
            self.igen = igen

    def make_hidden(self, batchsize):
        return self.tgen.make_hidden(batchsize)

    @property
    def n_frames(self):
        return self.tgen.n_frames

    def generate_video(self, z_slow, z_fast, c=None):
        """Generate videos from z_slow and z_fast.

        Args:
            z_slow (ndarray): The shape is (N, z_slow_dim).
            z_fast (ndarray): The shape is (N, n_frames, z_fast_dim).

        Returns:
            fake_video (ndarray): The shape is (N, out_channels, n_frames, H, W).
                H and W are determined by the output of ``video_gen`` layer.

        """
        assert(len(z_slow) == len(z_fast))
        N, n_frames, z_fast_dim = z_fast.shape
        z_fast = F.reshape(z_fast, (N * n_frames, z_fast_dim))

        _, z_slow_dim = z_slow.shape
        z_slow = F.reshape(
            F.broadcast_to(
                F.reshape(z_slow, (N, 1, z_slow_dim)),
                (N, n_frames, z_slow_dim)),
            (N * n_frames, z_slow_dim))
        if c is not None:
            c = F.reshape(
                F.broadcast_to(F.reshape(c, (N, 1)), (N, n_frames)),
                (N * n_frames,))

        fake_video = self.igen(z_slow, z_fast, c)
        _, n_ch, h, w = fake_video.shape
        fake_video = F.transpose(
            F.reshape(fake_video, (N, n_frames, n_ch, h, w)),
            [0, 2, 1, 3, 4])
        return fake_video

    def __call__(self, z_slow, c=None, n_frames=None):
        if 0 < self.n_classes and c is None:
            N = len(z_slow)
            c = self.xp.random.randint(low=0, high=self.n_classes, size=N)
            c = chainer.Variable(c.astype(self.xp.int32))
        n_frames = n_frames if n_frames is not None else n_frames
        z_fast = self.tgen(z_slow, c=c, n_frames=n_frames)
        return self.generate_video(z_slow, z_fast, c=c)
