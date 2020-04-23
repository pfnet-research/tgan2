import numpy
import scipy.linalg

import chainer
import chainer.cuda
from chainer import Variable


def get_mean_cov(classifier, samples, batchsize=16):
    '''Compute mean and covariance of dataset.'''
    N = len(samples)
    xp = classifier.xp

    ys = None
    for start in range(0, N, batchsize):
        end = min(start + batchsize, N)

        batch = samples[start:end]
        batch = Variable(xp.asarray(batch))  # To GPU if using CuPy

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = classifier.get_feature(batch)
        n_features = numpy.prod(y.shape[1:])
        if ys is None:
            ys = xp.empty((N, n_features), dtype=xp.float64)
        ys[start:end] = y.data.reshape(len(y.data), n_features)

    # Compute ean and covariance
    mean = chainer.cuda.to_cpu(xp.mean(ys, axis=0))
    # cov = F.cross_covariance(ys, ys, reduce='no').data.get()
    cov = numpy.cov(chainer.cuda.to_cpu(ys).T)

    return mean, cov


def get_FID(m0, c0, m1, c1):
    ret = 0
    ret += numpy.sum((m0 - m1) ** 2)
    ret += numpy.trace(c0 + c1 - 2.0 * scipy.linalg.sqrtm(numpy.dot(c0, c1)))
    return numpy.real(ret)


def make_FID_extension(gen, classifier, stat_file, batchsize=100, n_samples=1000):
    '''Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500'''

    @chainer.training.make_extension()
    def evaluation(trainer):
        stat = numpy.load(stat_file)

        xs = None
        for start in range(0, n_samples, batchsize):
            end = min(start + batchsize, n_samples)
            n = end - start
            z = gen.make_hidden(n)
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = gen(z)
            x = chainer.cuda.to_cpu(x.data)
            x = numpy.clip(x, -1, 1)
            if xs is None:
                xs = numpy.empty((n_samples,), x.shape[1:], dtype=numpy.float32)
            xs[start:end] = x

        mean, cov = get_mean_cov(classifier, xs, batchsize)
        fid = get_FID(stat['mean'], stat['cov'], mean, cov)
        chainer.reporter.report({'FID': fid})

    return evaluation
