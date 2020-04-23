#!/usr/bin/env python3

import argparse
import os

import chainer
import chainer.cuda
import numpy
from chainer import Variable

import cv2
import h5py
import imageio
import pandas

import tgan2
import yaml
from tgan2 import C3DVersion1UCF101
from tgan2 import UCF101Dataset
from tgan2.evaluations import fid
from tgan2.evaluations import inception_score
from tgan2.utils import make_instance


# len(dset) * n_loops == 9537 * 10 == 5610 * 17
def get_mean_cov(classifier, dataset, batchsize=17, n_iterations=5610):
    N = len(dataset)
    xp = classifier.xp

    ys = []
    it = chainer.iterators.MultiprocessIterator(
        dataset, batch_size=batchsize, shuffle=False, repeat=True, n_processes=8)
    for i, batch in enumerate(it):
        if i == n_iterations:
            break
        print('Compute {} / {}'.format(i + 1, n_iterations))

        batch = chainer.dataset.concat_examples(batch)
        batch = Variable(xp.asarray(batch))  # To GPU if using CuPy

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = classifier.get_feature(batch)
        n_features = numpy.prod(y.shape[1:])
        ys.append(chainer.cuda.to_cpu(y.data.reshape(len(y.data), n_features)))

    # Compute mean and covariance
    ys = numpy.concatenate(ys)
    mean = numpy.mean(ys, axis=0)
    cov = numpy.cov(ys.T)
    return mean.astype(numpy.float32), cov.astype(numpy.float32)


def main():
    parser = argparse.ArgumentParser()

    # For calculating statistics as the preparation
    parser.add_argument(
        '--ucf101-h5path-train', type=str,
        default='datasets/ucf101_192x256/train.h5')
    parser.add_argument(
        '--ucf101-config-train', type=str,
        default='datasets/ucf101_192x256/train.json')
    parser.add_argument(
        '--c3d-pretrained-model', type=str,
        default='datasets/models/conv3d_deepnetA_ucf.npz')
    parser.add_argument(
        '--stat-output', '-o', type=str,
        default='datasets/ucf101_192x256/ucf101_192x256_stat.npz')
    parser.add_argument('--test', action='store_true', default=False)

    # For calculating FID
    parser.add_argument('--stat-filename', '-s', type=str, default=None)
    parser.add_argument(
        '--ucf101-h5path-test', type=str,
        default='datasets/ucf101_192x256/test.h5')
    parser.add_argument(
        '--ucf101-config-test', type=str,
        default='datasets/ucf101_192x256/test.json')
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--gen-snapshot', '-g', type=str)
    parser.add_argument('--n-samples', '-n', type=int, default=2048)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--n-frames', '-f', type=int, default=16)
    parser.add_argument('--result-dir', '-d', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-loops', type=int, default=10)

    args = parser.parse_args()

    if args.test:
        stat = numpy.load(args.stat_filename)
        fid_result = fid.get_FID(stat['mean'], stat['cov'], stat['mean'], stat['cov'])
        print('FID:', fid_result)
        exit()

    if args.stat_filename is None:
        print('Loading')
        dataset = UCF101Dataset(
            n_frames=args.n_frames,
            h5path=args.ucf101_h5path_train,
            config_path=args.ucf101_config_train,
            img_size=192,
            stride=1)

        classifier = C3DVersion1UCF101(pretrained_model=args.c3d_pretrained_model)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            classifier.to_gpu()
        mean, cov = get_mean_cov(classifier, dataset)
        numpy.savez(args.stat_output, mean=mean, cov=cov)
    else:
        config = yaml.load(open(args.config))
        print(yaml.dump(config, default_flow_style=False))

        gen = make_instance(
            tgan2, config['gen'],
            args={'out_channels': 3})
        chainer.serializers.load_npz(args.gen_snapshot, gen)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            gen.to_gpu()
        
        classifier = C3DVersion1UCF101(pretrained_model=args.c3d_pretrained_model)
        if args.gpu >= 0:
            classifier.to_gpu()
   
        scores = []
        for i in range(args.n_loops):
            xs = inception_score.make_samples(
                gen,
                batchsize=args.batchsize,
                n_samples=args.n_samples,
                n_frames=args.n_frames)

            stat = numpy.load(args.stat_filename)
            mean, cov = fid.get_mean_cov(classifier, xs, args.batchsize)
            fid_result = fid.get_FID(stat['mean'], stat['cov'], mean, cov)
            print('{}\tFID:'.format(i), fid_result)
            
            scores.append(fid_result)

        mean, std = float(numpy.mean(scores)), float(numpy.std(scores))
        print('FID mean ({} times):\t'.format(args.n_loops), mean)
        print('FID stddev ({} times):\t'.format(args.n_loops), std)

        result = {'mean': mean, 'std': std}
        open(os.path.join(args.result_dir, 'FID.yml'), 'w').write(
                yaml.dump(result, default_flow_style=False))
        
if __name__ == '__main__':
    main()
