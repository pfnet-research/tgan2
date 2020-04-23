import logging
import math

import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np


logger = logging.getLogger(__name__)


def inception_score(classifier, samples, batchsize=100, splits=10, eps=1e-20):
    """Compute the inception score for given images.

    Default batchsize is 100 and split size is 10. Please refer to the
    official implementation. It is recommended to to use at least 50000
    images to obtain a reliable score.

    Reference:
    https://github.com/openai/improved-gan/blob/master/inception_score/classifier.py

    """
    n = len(samples)
    n_batches = int(math.ceil(float(n) / float(batchsize)))

    xp = classifier.xp

    # print('Batch size:', batchsize)
    # print('Total number of images:', n)
    # print('Total number of batches:', n_batches)

    # Compute the softmax predicitions for for all images, split into batches
    # in order to fit in memory

    ys = None  # Softmax container
    for i in range(n_batches):
        logger.info('Running batch %i/%i...', i + 1, n_batches)

        batch_start = (i * batchsize)
        batch_end = min((i + 1) * batchsize, n)

        samples_batch = samples[batch_start:batch_end]
        samples_batch = xp.asarray(samples_batch)  # To GPU if using CuPy
        samples_batch = Variable(samples_batch)

        # Feed images to the inception module to get the softmax predictions
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = F.softmax(classifier(samples_batch))
        if ys is None:
            n_class = y.shape[1]
            ys = xp.empty((n, n_class), dtype=xp.float64)
        ys[batch_start:batch_end] = y.data

    # Compute the inception score based on the softmax predictions of the
    # inception module.
    scores = xp.empty((splits), dtype=xp.float64)  # Split inception scores
    for i in range(splits):
        part = ys[(i * n // splits):((i + 1) * n // splits), :]
        part = part + eps  # to avoid convergence
        kl = part * (xp.log(part) -
                     xp.log(xp.expand_dims(xp.mean(part, 0), 0)))
        kl = xp.mean(xp.sum(kl, 1))
        scores[i] = xp.exp(kl)

    return xp.mean(scores), xp.std(scores)


def make_samples(gen, batchsize=100, n_samples=1000, n_frames=None):
    xs = None
    for start in range(0, n_samples, batchsize):
        end = min(start + batchsize, n_samples)
        n = end - start
        z = gen.make_hidden(n)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            args = {} if n_frames is None else {'n_frames': n_frames}
            x = gen(z, **args)
        x = chainer.cuda.to_cpu(x.data)
        x = np.clip(x, -1, 1)
        if xs is None:
            xs = np.empty((n_samples,) + x.shape[1:], dtype=np.float32)
        xs[start:end] = x
    return xs


def make_inception_score_extension(gen, classifier, batchsize=100, n_samples=1000, n_frames=None, splits=10):

    @chainer.training.make_extension()
    def evaluation(trainer):
        xs = make_samples(gen, batchsize=batchsize, n_samples=n_samples, n_frames=n_frames)
        mean, std = inception_score(classifier, xs, batchsize, splits)
        chainer.reporter.report({'IS_mean': mean, 'IS_std': std})

    return evaluation
