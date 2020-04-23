#!/usr/bin/env python3

import argparse
import warnings

import chainer
import numpy
import yaml

import tgan2
import tgan2.evaluations.inception_score
from tgan2.utils import make_config
from tgan2.utils import make_instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infiles', nargs='+', type=argparse.FileType('r'), default=())
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-m', '--model', default='arg.npz')
    parser.add_argument('-o', '--out', default='result.yml')
    parser.add_argument('--n-samples', type=int, default=2048)
    parser.add_argument('--n-loops', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--n-frames', type=int, default=16)
    args = parser.parse_args()

    conf_dicts = [yaml.load(fp) for fp in args.infiles]
    config = make_config(conf_dicts, args.attrs)
    return config, args


def main(config, args):
    print('Prepare model: {}'.format(args.model))
    gen = make_instance(
        tgan2, config['gen'], args={'out_channels': 3})
    chainer.serializers.load_npz(args.model, gen)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()

    conf_classifier = config['inception_score']['classifier']
    classifier = make_instance(tgan2, conf_classifier)
    if 'model_path' in conf_classifier:
        chainer.serializers.load_npz(
            conf_classifier['model_path'],
            classifier, path=conf_classifier['npz_path'])
    if args.gpu >= 0:
        classifier.to_gpu()

    scores = []
    for i in range(args.n_loops):
        print('Loop {}'.format(i))
        xs = tgan2.evaluations.inception_score.make_samples(
            gen, batchsize=args.batchsize,
            n_samples=args.n_samples, n_frames=args.n_frames)
        mean, std = tgan2.evaluations.inception_score.inception_score(
            classifier, xs, args.batchsize, splits=1)
        print(f'{mean} +- {std}')
        scores.append(chainer.backends.cuda.to_cpu(mean))

    scores = numpy.asarray(scores)
    mean, std = float(numpy.mean(scores)), float(numpy.std(scores))
    print(mean, std)

    result = {'mean': mean, 'std': std}
    open(args.out, 'w').write(yaml.dump(result, default_flow_style=False))



if __name__ == '__main__':
    config, args = parse_args()
    # Ignore warnings
    warnings.simplefilter('ignore')
    main(config, args)
