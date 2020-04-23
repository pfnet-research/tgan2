import chainer
import chainer.functions as F
import chainer.links as L

from tgan2.utils import make_batch_normalization


class ImageGenerator(chainer.Chain):

    """Generate frames from z_slow and z_fast arrays.

    This block has 5 deconvolution layers, and 4 layers out of the 5 will upsample the input to twice larger
    feature maps. So eventually the output will be  2 ** 4 = 16 times larger size of (bottom_width, bottom_width).

    Args:
        z_slow_dim (int): The dim of the first input array z_slow.
        z_fast_dim (int): The dim of the second input array z_fast.
        out_channels (int): The num of channels of generated videos.
        bottom_width (int): The shape of array which input z_slow and z_fast are reshaped to.
        ch (int): The num of channels of deconvolution 2d layers.

    """

    def __init__(self, z_slow_dim, z_fast_dim, out_channels, bottom_width,
                 ch=32, activation='relu', n_classes=0):
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim
        self.out_channels = out_channels
        self.bottom_width = bottom_width
        self.ch = ch
        self.activation = getattr(F, activation)
        self.n_classes = n_classes
        slow_mid_dim = bottom_width * bottom_width * ch * 8
        fast_mid_dim = bottom_width * bottom_width * ch * 8
        super(ImageGenerator, self).__init__()
        w = None
        with self.init_scope():
            self.l0s = L.Linear(z_slow_dim, slow_mid_dim, initialW=w, nobias=True)
            self.l0f = L.Linear(z_fast_dim, fast_mid_dim, initialW=w, nobias=True)
            self.dc1 = L.Deconvolution2D(ch * 16, ch * 8, 4, 2, 1, initialW=w, nobias=True)
            self.dc2 = L.Deconvolution2D(ch * 8, ch * 4, 4, 2, 1, initialW=w, nobias=True)
            self.dc3 = L.Deconvolution2D(ch * 4, ch * 2, 4, 2, 1, initialW=w, nobias=True)
            self.dc4 = L.Deconvolution2D(ch * 2, ch, 4, 2, 1, initialW=w, nobias=True)
            self.dc5 = L.Deconvolution2D(ch, out_channels, 3, 1, 1, initialW=w, nobias=False)

            self.bn0s = make_batch_normalization(slow_mid_dim, n_classes=n_classes)
            self.bn0f = make_batch_normalization(fast_mid_dim, n_classes=n_classes)
            self.bn1 = make_batch_normalization(ch * 8, n_classes=n_classes)
            self.bn2 = make_batch_normalization(ch * 4, n_classes=n_classes)
            self.bn3 = make_batch_normalization(ch * 2, n_classes=n_classes)
            self.bn4 = make_batch_normalization(ch, n_classes=n_classes)

    def __call__(self, z_slow, z_fast, c=None):
        """Generate a frame of a video from z_slow and z_fast.

            The shape of inputs are:

        Args:
            z_slow (ndarray): The shape is (N, z_slow_dim).
            z_fast (ndarray): The shape is (N, n_frames, z_fast_dim).

        Returns:
            video (ndarray): The shape is (N, out_channels, bottom_width * 4, bottom_width * 4)

        """
        if self.n_classes == 0:
            assert(c is None)
        else:
            assert(c is not None)

        def apply_bn(func, h):
            return func(h, c) if c is not None else func(h)

        n = z_slow.shape[0]
        h_slow = F.reshape(F.relu(self.bn0s(self.l0s(z_slow))), (n, self.ch * 8, self.bottom_width, self.bottom_width))
        h_fast = F.reshape(F.relu(self.bn0f(self.l0f(z_fast))), (n, self.ch * 8, self.bottom_width, self.bottom_width))
        h = F.concat([h_slow, h_fast], axis=1)
        h = self.activation(apply_bn(self.bn1, self.dc1(h)))
        h = self.activation(apply_bn(self.bn2, self.dc2(h)))
        h = self.activation(apply_bn(self.bn3, self.dc3(h)))
        h = self.activation(apply_bn(self.bn4, self.dc4(h)))
        x = F.tanh(self.dc5(h))
        return x
