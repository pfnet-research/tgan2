import chainer
import numpy as np


class MovingMNISTDataset(chainer.dataset.DatasetMixin):

    """Moving MNIST Dataset

    Args:
        n_frames (int): The number of frames which the returned array has in channel dimension.
            Typically it's 16.
        dataset_path (str): The .npy file of moving MNIST dataset.

    Returns:
        ndarray: The shape is (N, n_frames, 64, 64)

    """

    def __init__(self, n_frames, dataset_path):
        self.dset = np.load(dataset_path)
        self.dset = self.dset.transpose([1, 0, 2, 3])
        self.dset = self.dset[:, np.newaxis, :, :, :]
        self.n_frames = n_frames
        self.T = self.dset.shape[2]

    def __len__(self):
        return self.dset.shape[0]

    @property
    def n_channels(self):
        return 1

    def get_example(self, i):
        ot = np.random.randint(self.T - self.n_frames) \
            if self.T > self.n_frames else 0
        x = self.dset[i, :, ot:(ot + self.n_frames), :, :]
        return np.asarray((x - 128.0) / 128.0, dtype=np.float32)
