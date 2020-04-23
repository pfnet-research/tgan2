from tgan2.datasets.h5video import HDF5VideoDataset


class UCF101Dataset(HDF5VideoDataset):

    def __init__(self, n_frames, h5path, config_path, img_size,
                 label=False, stride=1, xmargin=4, xflip=True, n_samples_per_label=None):
        super(UCF101Dataset, self).__init__(
            n_frames, h5path, config_path, img_size, label, stride, xmargin, xflip)

        if n_samples_per_label is not None:
            self.conf = self.conf.groupby('category').apply(
                lambda x: x.sample(
                    n=n_samples_per_label,
                    random_state=0)).reset_index(drop=True)

    @property
    def n_channels(self):
        return 3

    @property
    def n_classes(self):
        return 101
