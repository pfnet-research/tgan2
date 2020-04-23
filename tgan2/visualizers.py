import os

import chainer
import chainer.cuda
import imageio
import numpy as np
from PIL import Image


def out_generated_movie(gen, dis, rows, cols, seed, dst, batchsize=16, n_frames=None):
    @chainer.training.make_extension()
    def make_image(trainer):
        xp = gen.xp
        xp.random.seed(seed)

        N = rows * cols
        xs = generate_video(gen, N, seed, batchsize, n_frames)

        preview_dir = os.path.join(dst, 'preview')
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        preview_path = os.path.join(
            preview_dir, 'image{:0>8}.png'.format(trainer.updater.iteration))
        video_mp4_path = os.path.join(
            preview_dir, 'video{:0>8}.mp4'.format(trainer.updater.iteration))

        save_image(xs, preview_path)
        save_video(xs, rows, cols, video_mp4_path)
    return make_image


def generate_video(gen, N, seed=None, batchsize=16, n_frames=None):
    if seed is not None:
        xp = gen.xp
        xp.random.seed(seed)

    xs = None
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        for start in range(0, N, batchsize):
            end = min(start + batchsize, N)
            n = end - start
            z_slow = gen.make_hidden(n)
            args = {} if n_frames is None else {'n_frames': n_frames}
            x_fake = gen(z_slow, **args)
            x_fake = chainer.cuda.to_cpu(x_fake.data)
            x_fake = np.clip(x_fake * 128.0 + 128.0, 0, 255).astype(np.uint8)
            if xs is None:
                xs = np.empty((N,) + x_fake.shape[1:], dtype=np.uint8)
            xs[start:end] = x_fake
    return xs


def save_image(x, dst_path, mode=None):
    N, C, F, H, W = x.shape
    x = x.reshape((N, C, F, H, W))
    x = x.transpose(0, 3, 2, 4, 1)
    if C == 1:
        x = x.reshape((N * H, F * W))
    else:
        x = x.reshape((N * H, F * W, C))

    image = Image.fromarray(x, mode=mode).convert('RGB')
    image.save(dst_path)


def save_video(x, rows, cols, dst_path):
    N, C, F, H, W = x.shape
    x = x.reshape((rows, cols, C, F, H, W))
    x = x.transpose(3, 0, 4, 1, 5, 2)
    if C == 1:
        x = x.reshape((F, rows * H, cols * W))
    else:
        x = x.reshape((F, rows * H, cols * W, C))

    writer = imageio.get_writer(dst_path, fps=15)
    for t in range(F):
        writer.append_data(x[t].astype(np.uint8))
    writer.close()
