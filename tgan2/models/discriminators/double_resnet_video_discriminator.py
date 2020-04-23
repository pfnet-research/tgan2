import chainer
import numpy as np

from tgan2.models.discriminators.resnet_frame_discriminator import ResNetFrameDiscriminator
from tgan2.models.discriminators.resnet_video_discriminator import ResNetVideoDiscriminator


class DoubleResNetVideoDiscriminator(chainer.Chain):

    def __init__(self, in_channels, mid_ch=64, n_classes=0, activation='relu'):
        super(DoubleResNetVideoDiscriminator, self).__init__()
        self.levels = []
        with self.init_scope():
            self.frame_dis = ResNetFrameDiscriminator(in_channels, mid_ch, n_classes, activation)
            self.video_dis = ResNetVideoDiscriminator(in_channels, mid_ch, n_classes, activation)

    def __call__(self, x, c=None):
        y_video = self.video_dis(x, c)
        t = np.random.randint(x.shape[2])
        y_frame = self.frame_dis(x[:, :, t], c)
        return y_video + y_frame
