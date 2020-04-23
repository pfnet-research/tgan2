import argparse

import chainer
import yaml

import tgan2
from tgan2.visualizers import generate_video
from tgan2.visualizers import save_video
from tgan2.utils import make_config
from tgan2.utils import make_instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infiles', nargs='+', type=argparse.FileType('r'), default=())
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-m', '--model', default='arg.npz')
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--cols', type=int, default=6)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('-o', '--out', default='out.mp4')
    args = parser.parse_args()

    conf_dicts = [yaml.load(fp) for fp in args.infiles]
    config = make_config(conf_dicts, args.attrs)
    return config, args


def main(config, args):
    gen = make_instance(
        tgan2, config['gen'], args={'out_channels': 3})
    chainer.serializers.load_npz(args.model, gen)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()

    N = args.rows * args.cols
    video = tgan2.visualizers.generate_video(
        gen, N, seed=0, batchsize=args.batchsize, n_frames=args.frames)
    save_video(video, args.rows, args.cols, args.out)


if __name__ == '__main__':
    config, args = parse_args()
    main(config, args)
