import chainer
import chainer.functions as F

from tgan2.models.discriminators.resnet_video_discriminator import ResNetVideoDiscriminator


class MultiResNetVideoDiscriminator(chainer.Chain):

    def __init__(self, in_channels, mid_ch=64, n_classes=0, activation='relu', n_levels=4):
        super(MultiResNetVideoDiscriminator, self).__init__()
        self.levels = []
        with self.init_scope():
            for i in range(n_levels):
                level = ResNetVideoDiscriminator(
                    in_channels, mid_ch, n_classes, activation)
                setattr(self, 'level{}'.format(i), level)
                self.levels.append(level)

    def __call__(self, xs, c=None):
        assert(len(xs) == len(self.levels))
        outputs = [level(x, c) for x, level in zip(xs, self.levels)]
        output = F.concat(outputs, axis=0)
        return output
