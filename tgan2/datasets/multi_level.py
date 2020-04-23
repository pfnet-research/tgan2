import chainer
import numpy

import tgan2
from tgan2.utils import make_instance


def _expand(option):
    if isinstance(option, (list, tuple)):
        return option
    elif option:
        return [2, 2, 2]
    else:
        return [1, 1, 1]


def subsample(x, stride=2):
    offset = numpy.random.randint(stride)
    return x[:, offset::stride]


def pooling(x, ksize):
    if ksize == 1:
        return x
    C, T, H, W = x.shape
    Hd = H // ksize
    Wd = W // ksize
    return x.reshape(C, T, Hd, ksize, Wd, ksize).mean(axis=(3, 5))


class MultiLevelDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, subsample_frame=True):
        self.subsample_frame = _expand(subsample_frame)
        if isinstance(dataset, dict):
            self.dataset = make_instance(tgan2, dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @property
    def n_channels(self):
        return self.dataset.n_channels

    @property
    def n_classes(self):
        return self.dataset.n_classes

    def get_example(self, i):
        x = self.dataset.get_example(i)
        x1 = x
        x2 = subsample(x1, stride=self.subsample_frame[0])
        x3 = subsample(x2, stride=self.subsample_frame[1])
        x4 = subsample(x3, stride=self.subsample_frame[2])
        x1 = pooling(x1, ksize=8)
        x2 = pooling(x2, ksize=4)
        x3 = pooling(x3, ksize=2)
        x4 = pooling(x4, ksize=1)
        return x1, x2, x3, x4
