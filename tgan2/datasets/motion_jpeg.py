import logging
import os

from chainer.dataset import dataset_mixin
import numpy
import pandas
from PIL import Image


logger = logging.getLogger(__name__)


class MotionJPEGDataset(dataset_mixin.DatasetMixin):

    def __init__(self, src_dir, conf_path,
                 n_frames=16, src_size=(128, 128),
                 dst_size=(64, 64), label=True):
        self.src_dir = src_dir
        self.conf = pandas.read_json(conf_path, dtype=False)
        self.n_frames = 16
        self.src_size = src_size
        self.dst_size = dst_size
        self.label = label

    @property
    def n_channels(self):
        return 3

    def __len__(self):
        return len(self.conf)

    def resize(self, img):
        src_W, src_H = img.size
        assert(src_H == self.src_size[0])
        T = src_W // self.src_size[1]
        dst_W = self.dst_size[1] * T
        dst_H = self.dst_size[0]
        return img.resize(size=(dst_W, dst_H))

    def get_example(self, i):
        row = self.conf.iloc[i]
        image_path = os.path.join(self.src_dir, row['filepath'])
        image = Image.open(image_path)
        image = numpy.asarray(
            self.resize(image).convert('RGB'), dtype=numpy.float32)

        H = image.shape[0]
        W = self.dst_size[1]
        T = image.shape[1] // W
        C = image.shape[2]
        mov = image.reshape(H, T, W, C)

        start = numpy.random.randint(0, T - self.n_frames)
        end = start + self.n_frames
        mov = mov[:, start:end].transpose(3, 1, 0, 2)
        mov = (mov - 127.5) / 127.5
        if self.label:
            label = numpy.int32(row['label'])
            return mov, label
        else:
            return mov

    @classmethod
    def make_dataset(cls, frame, src_dir='.', dst_dir='.', size=(128, 128)):
        from sklearn.externals import joblib

        success_flag = joblib.Parallel(n_jobs=10)(
            joblib.delayed(make_frame)(
                src_dir, x['filepath_mov'], dst_dir, x['filepath'], size)
            for (ind, x) in frame.iterrows())
        dst_frame = frame[success_flag]
        return dst_frame


def make_frame(src_dir, filepath_mov, dst_dir, filepath, size):
    import imageio
    src_path = os.path.join(src_dir, filepath_mov)
    try:
        reader = imageio.get_reader(src_path)
        images = []
        for img in reader:
            img2 = numpy.asarray(
                Image.fromarray(img).resize(size).convert('RGB'))
            images.append(img2)
    except (RuntimeError, OSError):
        logger.warning('Raise error when reading %s', filepath_mov)
        return False

    H, W = images[0].shape[:2]
    T = len(images)
    C = 3
    mov = numpy.concatenate(images).reshape(T, H, W, C)
    mov = mov.transpose(1, 0, 2, 3)
    mov = mov.reshape(H, T * W, C)

    dst_path = os.path.join(dst_dir, filepath)
    target_dir = os.path.dirname(dst_path)
    os.makedirs(target_dir, exist_ok=True)
    Image.fromarray(mov).save(dst_path)
    return True


def init_logger(filename='', level=logging.INFO):
    kwargs = {
        'format': '%(asctime)s [%(levelname)s] %(message)s',
        'level': level,
    }
    if filename != '':
        kwargs['filename'] = filename
    logging.basicConfig(**kwargs)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


if __name__ == '__main__':
    src_dir = '/mnt/sakuradata10-striped/datasets/moments/Moments_in_Time_256x256_30fps/training'
    src_csv = '/mnt/sakuradata10-striped/datasets/moments/Moments_in_Time_256x256_30fps/trainingSet.csv'
    src_category_csv = '/mnt/sakuradata10-striped/datasets/moments/Moments_in_Time_256x256_30fps/moments_categories.txt'
    dst_dir = '/mnt/sakuradata10-striped/msaito/datasets/Moments_in_Time_128x128_jpeg/training'
    dst_json = '/mnt/sakuradata10-striped/msaito/datasets/Moments_in_Time_128x128_jpeg/training.json'

    logger = init_logger()

    logger.info('Loading dataset')
    src_frame = pandas.read_csv(src_csv, header=None)
    src_frame = src_frame.head(n=10000)
    categories = pandas.read_csv(src_category_csv, header=None)
    category_to_id = {row[0]: row[1] for ind, row in categories.iterrows()}

    logger.info('Making frame')
    frame = pandas.DataFrame({'filepath_mov': src_frame[0]})
    frame['category'] = frame.apply(lambda x: os.path.dirname(x['filepath_mov']), axis=1)
    frame['label'] = frame.apply(lambda x: category_to_id[x['category']], axis=1)
    frame['filepath'] = frame.apply(lambda x: os.path.splitext(x['filepath_mov'])[0] + '.jpg', axis=1)

    logger.info('Making dataset')
    frame = MotionJPEGDataset.make_dataset(frame, src_dir, dst_dir)

    logger.info('Making frame')
    frame.to_json(dst_json)
