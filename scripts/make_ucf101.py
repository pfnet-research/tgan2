import argparse
import os
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue

import cv2
import h5py
import imageio
import numpy
import pandas
from tqdm import tqdm


def process_video(in_queue, out_queue):
    while True:
        ret = in_queue.get()
        if ret is None:
            break
        video_path, img_rows, img_cols, row = ret

        try:
            video_reader = imageio.get_reader(video_path)
        except Exception as e:
            print(e)
            print(video_path)
            continue

        video = []
        while True:
            try:
                img = video_reader.get_next_data()
            except IndexError:
                break
            else:
                dst_img = cv2.resize(
                    img, (img_cols, img_rows),
                    interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
                video.append(dst_img)
        T = len(video)
        video = numpy.concatenate(video).reshape(T, 3, img_rows, img_cols)
        video_reader.close()
        out_queue.put((video, row))


def make_frame(filepath):
    frame = pandas.read_csv(
        filepath, sep=' ', header=None, names=['filename', 'label'])
    del frame['label']
    frame['filename'] = frame['filename'].apply(lambda x: os.path.basename(x))
    frame['category'] = frame['filename'].apply(lambda x: x.split('_')[1])
    return frame


def count_frames(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print(e)
        print(path)
        return 0

    n_frames = 0
    while True:
        try:
            reader.get_next_data()
        except IndexError:
            break
        else:
            n_frames += 1
    reader.close()
    return n_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', type=str, default='datasets/ucf101', help='')
    parser.add_argument('--src-split-dir', type=str, default='datasets/ucf101/ucfTrainTestlist', help='')
    parser.add_argument('--dst-dir', type=str, default='datasets/ucf101', help='')
    parser.add_argument('--img-rows', type=int, default=192)
    parser.add_argument('--img-cols', type=int, default=256)
    parser.add_argument('--n-frames', type=int, default=None)
    args = parser.parse_args()

    out_dir = f'{args.dst_dir}_{args.img_rows}x{args.img_cols}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for name in ['train', 'test']:
        print('Processing {} dataset'.format(name))
        path = os.path.join(args.src_split_dir, '{}list01.txt'.format(name))
        frame = make_frame(path)

        if args.n_frames is None:
            print('Count # of frames')
            p = Pool()
            rets = []
            for ind, row in frame.iterrows():
                path = os.path.join(args.src_dir, row['filename'])
                rets.append(p.apply_async(count_frames, args=(path,)))
            n_frames = 0
            for ret in tqdm(rets):
                n_frames += ret.get()
        else:
            n_frames = args.n_frames
        print('# of frames: {}'.format(n_frames))

        h5file = h5py.File(os.path.join(out_dir, '{}.h5'.format(name)), 'w')
        dset = h5file.create_dataset(
            'image', (n_frames, 3, args.img_rows, args.img_cols), dtype=numpy.uint8)

        in_queue = Queue()
        out_queue = Queue(maxsize=os.cpu_count())
        processes = [Process(target=process_video, args=(in_queue, out_queue))
                     for _ in range(os.cpu_count())]
        for p in processes:
            p.start()
        for ind, row in frame.iterrows():
            path = os.path.join(args.src_dir, row['filename'])
            in_queue.put((path, args.img_rows, args.img_cols, row))
        for _ in range(len(processes)):
            in_queue.put(None)

        conf = []
        start = 0
        for i in tqdm(range(len(frame))):
            video, row = out_queue.get()
            T = len(video)
            dset[start:(start + T)] = video
            conf.append({
                'start': start,
                'end': (start + T),
                'category': row['category']
            })
            start += T

        for p in processes:
            p.join()

        conf = pandas.DataFrame(conf)
        conf.to_json(os.path.join(out_dir, '{}.json'.format(name)), orient='records')
        h5file.close()
