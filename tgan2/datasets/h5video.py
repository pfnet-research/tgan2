import chainer
import h5py
import numpy as np
import pandas as pd


class HDF5VideoDataset(chainer.dataset.DatasetMixin):

    def __init__(self, n_frames, h5path, config_path, img_size,
                 label=False, stride=1, xmargin=0, xflip=False):
        self.n_frames = n_frames
        self.h5path = h5path
        self.h5file = None
        self.dset = None

        self.conf = pd.read_json(config_path)
        self.img_size = img_size
        self.label = label
        self.stride = stride
        self.xmargin = xmargin
        self.xflip = xflip
        if self.label:
            category = self.conf['category'].unique()
            category.sort()
            self.category_to_index = {k: i for (i, k) in enumerate(category)}

    @property
    def n_channels(self):
        return 3

    @property
    def n_classes(self):
        if self.label:
            return len(self.category_to_index)
        else:
            return 0

    def __len__(self):
        return len(self.conf)

    def _crop_center(self, mov):
        T, C, H, W = mov.shape

        y0 = (H - self.img_size) // 2
        if chainer.config.train:
            x0 = np.random.randint(
                self.xmargin, W - self.img_size + 1 - self.xmargin)
        else:
            x0 = (W - self.img_size) // 2
        mov = mov[:, :, y0:(y0 + self.img_size), x0:(x0 + self.img_size)]
        assert mov.shape[2] == self.img_size
        assert mov.shape[3] == self.img_size
        if self.xflip and chainer.config.train and np.random.rand() > 0.5:
            mov = mov[:, :, :, ::-1]
        return mov

    def get_example(self, i):
        if self.h5file is None and self.dset is None:
            self.h5file = h5py.File(self.h5path, 'r')
            self.dset = self.h5file['image']
        mov_info = self.conf.iloc[i]
        total_frames = mov_info.end - mov_info.start
        clip_frames = (self.n_frames - 1) * self.stride + 1
        start = mov_info.start + np.random.randint(0, total_frames - clip_frames)
        end = start + clip_frames
        assert(mov_info.start <= start < end <= mov_info.end)
        x = self.dset[start:end:self.stride]
        x = self._crop_center(x)
        x = x.transpose(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        x = np.asarray((x - 128.0) / 128.0, dtype=np.float32)

        if self.label:
            c = np.int32(self.category_to_index[mov_info.category])
            return x, c
        else:
            return x
