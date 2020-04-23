import csv
import os
import random

import chainer
import numpy
from chainer.dataset import dataset_mixin

try:
    import pynvvl
    pynvvl_enabled = True
except ImportError:
    pynvvl_enabled = False


class VideoDataset(dataset_mixin.DatasetMixin):

    def __init__(self, n_frames, size, root='.', video_path=None, label_path=None,
                 dtype=numpy.float32, label_dtype=numpy.int32,
                 include_label=True, stride=1, frame_offset=0):
        self.n_frames = n_frames
        self.size = size
        self.root = root
        self.dtype = dtype
        self.label_dtype = label_dtype
        self.include_label = include_label
        self.stride = stride
        self.frame_offset = frame_offset

        if label_path is not None:
            label2id = {}
            with open(label_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    label2id[row[0]] = label_dtype(row[1])

        with open(video_path) as f:
            dset = []
            reader = csv.reader(f)
            for row in reader:
                if include_label:
                    path, label = row[0], row[1]
                    if label_path is not None:
                        dset.append((path, label2id[label]))
                    else:
                        dset.append((path, label_dtype(label)))
                else:
                    dset.append((path,))

        self._dset = dset
        self._buffer = None
        self._device_id = None

    def __len__(self):
        return len(self._dset)

    def get_example(self, i):
        if not pynvvl_enabled:
            raise RuntimeError('pynvvl is not installed')

        if self._device_id is None:
            self._device_id = chainer.cuda.Device(device=None).id
        loader = pynvvl.NVVLVideoLoader(device_id=self._device_id)
        clip_frames = (self.n_frames - 1) * self.stride + 1
        if self._buffer is None:
            with chainer.cuda.Device(self._device_id):
                self._buffer = chainer.cuda.cupy.empty(
                    (clip_frames, 3, self.size, self.size),
                    dtype=numpy.float32)

        video_path = self._dset[i][0]
        full_path = os.path.join(self.root, video_path)

        total_frames = loader.frame_count(full_path)
        start = random.randint(
            self.frame_offset,
            total_frames - clip_frames - self.frame_offset - 1)
        video = loader.read_sequence(
            full_path, frame=start, count=clip_frames,
            scale_height=self.size, scale_width=self.size,
            crop_height=self.size, crop_width=self.size,
            horiz_flip=True, out=self._buffer)
        video = video[::self.stride, :, :, :]
        video = (video.transpose(1, 0, 2, 3) - 128) / 128

        if self.include_label:
            return video, self._dset[i][1]
        else:
            return video
