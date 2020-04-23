import itertools

import chainer
import numpy as np


class OperableMovingMNISTDataset(chainer.dataset.DatasetMixin):

    """Operable Moving MNIST Dataset

    Args:
        n_frames (int): The number of frames which the returned array has in channel dimension.
        rows (int): The height of the returned video.
        cols (int): The width of the returned video.
        n_digits (int or tuple): The number of digits the returned array contains.
            If tuple is specified, it randomly outputs multiple digits within a range of t[0] and t[1].
        label (bool): If True, it also outputs an array with length 10 where
            each element denotes the number of the digit.

    Returns:
        ndarray: The shape is (N, n_frames, rows, cols)

    """

    def __init__(self, n_frames=16, rows=64, cols=64, n_digits=2, label=False, train=True):
        self.n_frames = n_frames
        self.rows = rows
        self.cols = cols
        train, test = chainer.datasets.mnist.get_mnist()
        if train:
            self.dset = train
        else:
            self.dset = test
        self.n = len(self.dset)
        self.n_digits = n_digits
        self.label = label
        if label:
            digits = set(tuple(sorted(t)) for t in itertools.product(range(10), repeat=n_digits))
            self._digits_to_label = {x: i for i, x in enumerate(sorted(digits))}

    def __len__(self):
        return self.n

    @property
    def n_channels(self):
        return 1

    @property
    def n_classes(self):
        return len(self._digits_to_label)

    def make_trajectory(self, length, rows=36, cols=36):

        def make_position(rows, cols):
            return np.array([rows, cols]) * np.random.rand(2)

        def make_velocity():
            theta = (2 * np.pi) * np.random.rand()
            s_min, s_max = 1.0, 5.0
            s = s_min + (s_max - s_min) * np.random.rand()
            return np.array([s * np.cos(theta), s * np.sin(theta)])

        def proceed(p0, v0, size):
            p1 = p0 + v0
            v1 = v0
            if p1 < 0:
                p1 = -p1
                v1 = -v0
            elif size < p1:
                p1 = size - (p1 - size)
                v1 = -v0
            return p1, v1

        pos, vel = make_position(rows, cols), make_velocity()

        positions = np.zeros((length, 2), dtype=np.float64)
        velocities = np.zeros((length, 2), dtype=np.float64)
        positions[0] = pos
        velocities[0] = vel
        for t in range(1, length):
            p0, v0 = positions[t - 1], velocities[t - 1]
            p1y, v1y = proceed(p0[0], v0[0], rows)
            p1x, v1x = proceed(p0[1], v0[1], cols)
            positions[t, :] = [p1y, p1x]
            velocities[t, :] = [v1y, v1x]
        return positions.astype(np.int32), velocities.astype(np.int32)

    def draw(self, mov, pic):
        p, _ = self.make_trajectory(
            self.n_frames,
            rows=(self.rows - 28),
            cols=(self.cols - 28))
        for t in range(self.n_frames):
            y, x = p[t]
            mov[0, t, y:(y + 28), x:(x + 28)] += pic.reshape(28, 28)

    def make_label(self, labels):
        return self._digits_to_label[tuple(sorted(labels))]

    def get_example(self, i):
        mov = np.zeros((1, self.n_frames, self.rows, self.cols), dtype=np.float32)
        labels = []
        if isinstance(self.n_digits, int):
            n_digits = self.n_digits
        else:
            n_digits = np.random.randint(self.n_digits[0], self.n_digits[1])

        for s in range(n_digits):
            if s == 0:
                d, t = self.dset[i]
            else:
                j = np.random.randint(self.n)
                d, t = self.dset[j]
            self.draw(mov, d)
            labels.append(t)
        mov[mov > 1] = 1
        mov = 2 * mov - 1
        if self.label:
            return mov, self.make_label(labels)
        else:
            return mov
