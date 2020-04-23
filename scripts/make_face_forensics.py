import argparse
import os
from multiprocessing import Pool

import cv2
import h5py
import imageio
import numpy
import pandas
from tqdm import tqdm


def parse_videos(root):
    splits = ['train', 'test', 'val']
    categories = ['original', 'mask', 'altered']
    results = []
    for split in splits:
        for category in categories:
            target_dir = os.path.join(root, split, category)
            filenames = sorted(os.listdir(target_dir))
            for filename in filenames:
                results.append({
                    'split': split,
                    'category': category,
                    'filename': filename,
                    'filepath': os.path.join(split, category, filename),
                })
    return pandas.DataFrame(results)


def crop(img, left, right, top, bottom, margin):
    cols = right - left
    rows = bottom - top
    if cols < rows:
        padding = rows - cols
        left -= padding // 2
        right += (padding // 2) + (padding % 2)
        cols = right - left
    else:
        padding = cols - rows
        top -= padding // 2
        bottom += (padding // 2) + (padding % 2)
        rows = bottom - top
    assert(rows == cols)
    return img[top:bottom, left:right]


def process_video(video_path, mask_path, size, threshold=5, margin=0.02):
    video_reader = imageio.get_reader(video_path)
    mask_reader = imageio.get_reader(mask_path)
    assert(video_reader.get_length() == mask_reader.get_length())

    video = []
    for img, mask in zip(video_reader, mask_reader):
        hist = (255 - mask).astype(numpy.float64).sum(axis=2)
        horiz_hist = numpy.where(hist.mean(axis=0) > threshold)[0]
        vert_hist = numpy.where(hist.mean(axis=1) > threshold)[0]
        left, right = horiz_hist[0], horiz_hist[-1]
        top, bottom = vert_hist[0], vert_hist[-1]
        dst_img = crop(img, left, right, top, bottom, margin)
        dst_img = cv2.resize(
            dst_img, (size, size),
            interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
        video.append(dst_img)
    T = len(video)
    video = numpy.concatenate(video).reshape(T, 3, size, size)
    return video


def count_frames(path):
    reader = imageio.get_reader(path)
    n_frames = 0
    while True:
        try:
            img = reader.get_next_data()
        except IndexError as e:
            break
        else:
            n_frames += 1
    return n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='datasets/FaceForensics_compressed')
    parser.add_argument('--dst-dir', type=str, default='datasets/face256px')
    parser.add_argument('--img-size', type=int, default=256)
    args = parser.parse_args()

    frame = parse_videos(args.root)
    os.makedirs(args.dst_dir, exist_ok=True)

    p = Pool()
    for split in ['train', 'val', 'test']:
        print('Processing {}'.format(split))
        target_frame = frame[frame['split'] == split]
        filenames = target_frame['filename'].unique()

        print('Count # of frames')
        rets = []
        for i, filename in enumerate(filenames):
            fn_frame = target_frame[target_frame['filename'] == filename]
            video_path = os.path.join(
                args.root, fn_frame[fn_frame['category'] == 'original'].iloc[0]['filepath'])
            rets.append(p.apply_async(count_frames, args=(video_path,)))
        n_frames = 0
        for ret in tqdm(rets):
            n_frames += ret.get()
        print('# of frames: {}'.format(n_frames))

        h5file = h5py.File(os.path.join(args.dst_dir, '{}.h5'.format(split)), 'w')
        dset = h5file.create_dataset(
            'image', (n_frames, 3, args.img_size, args.img_size), dtype=numpy.uint8)
        conf = []
        start = 0
        for i, filename in enumerate(filenames):
            print('Processing {} / {}'.format(i, len(filenames)))
            fn_frame = target_frame[target_frame['filename'] == filename]
            video_path = os.path.join(
                args.root, fn_frame[fn_frame['category'] == 'original'].iloc[0]['filepath'])
            mask_path = os.path.join(
                args.root, fn_frame[fn_frame['category'] == 'mask'].iloc[0]['filepath'])
            video = process_video(video_path, mask_path, args.img_size)
            T = len(video)
            dset[start:(start + T)] = video
            conf.append({'start': start, 'end': (start + T)})
            start += T
        conf = pandas.DataFrame(conf)
        conf.to_json(os.path.join(args.dst_dir, '{}.json'.format(split)), orient='records')


if __name__ == '__main__':
    main()
