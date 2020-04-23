#!/usr/bin/env python3

import argparse
import copy
import hashlib
import os
import random
import warnings

import yaml

import chainer
import chainermn
import numpy
import tgan2
from chainer import cuda
from chainer import training
from chainer.training import extensions
from tgan2.utils import make_instance

cuda.set_max_workspace_size(1024 * 1024 * 1024)
chainer.global_config.autotune = True
chainer.config.comm = None


def get_device_communicator(gpu, communicator, seed, batchsize):
    if gpu:
        if communicator == 'naive':
            print('Error: \'naive\' communicator does not support GPU.\n')
            exit(-1)
        comm = chainermn.create_communicator(communicator)
        device = comm.intra_rank
    else:
        if communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        device = -1

    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(communicator))
        print('Using seed: {}'.format(seed))
        print('Entire batchsize: {}'.format(comm.mpi_comm.size * batchsize))
        print('==========================================')

    return device, comm


def train(config):
    config_backup = copy.deepcopy(config)

    # Setup
    device, comm = get_device_communicator(
        config['gpu'], config['communicator'],
        config['seed'], config['batchsize'])
    chainer.config.comm = comm  # To use from the inside of models

    if config.get('seed', None) is not None:
        random.seed(config['seed'])
        numpy.random.seed(config['seed'])
        cuda.cupy.random.seed(config['seed'])

    # Prepare dataset and models
    if not config['label']:
        if comm.mpi_comm.rank == 0:
            dataset = make_instance(tgan2, config['dataset'])
        else:
            dataset = None
        dataset = chainermn.scatter_dataset(dataset, comm, shuffle=True)
        # Retrieve property from the original of SubDataset
        n_channels = dataset._dataset.n_channels
        gen = make_instance(
            tgan2, config['gen'], args={'out_channels': n_channels})
        dis = make_instance(
            tgan2, config['dis'], args={'in_channels': n_channels})
    else:
        if comm.mpi_comm.rank == 0:
            print('## NOTE: Training Conditional TGAN')
            dataset = make_instance(tgan2, config['dataset'], args={'label': True})
        else:
            dataset = None
        dataset = chainermn.scatter_dataset(dataset, comm, shuffle=True)
        # Retrieve property from the original of SubDataset
        n_channels = dataset._dataset.n_channels
        n_classes = dataset._dataset.n_classes
        gen = make_instance(
            tgan2, config['gen'],
            args={'out_channels': n_channels, 'n_classes': n_classes})
        dis = make_instance(
            tgan2, config['dis'],
            args={'in_channels': n_channels, 'n_classes': n_classes})

    if device >= 0:
        chainer.cuda.get_device(device).use()
        gen.to_gpu()
        dis.to_gpu()

    if comm.mpi_comm.rank == 0:
        def print_params(link):
            n_params = sum([p.size for n, p in link.namedparams()])
            print('# of params in {}:\t{}'.format(
                link.__class__.__name__, n_params))
        print_params(gen)
        print_params(dis)

    # Prepare optimizers
    gen_optimizer = chainermn.create_multi_node_optimizer(
        make_instance(chainer.optimizers, config['gen_opt']), comm)
    dis_optimizer = chainermn.create_multi_node_optimizer(
        make_instance(chainer.optimizers, config['dis_opt']), comm)
    gen_optimizer.setup(gen)
    dis_optimizer.setup(dis)
    optimizers = {
        'generator': gen_optimizer, 'discriminator': dis_optimizer,
    }

    iterator = chainer.iterators.MultithreadIterator(
        dataset, batch_size=config['batchsize'])
    updater = make_instance(
        tgan2, config['updater'],
        args={'iterator': iterator, 'optimizer': optimizers, 'device': device})

    # Prepare trainer and its extensions
    trainer = training.Trainer(
        updater, (config['iteration'], 'iteration'), out=config['out'])
    snapshot_interval = (config['snapshot_interval'], 'iteration')
    display_interval = (config['display_interval'], 'iteration')

    if comm.rank == 0:
        # Inception score
        if config.get('inception_score', None) is not None:
            conf_classifier = config['inception_score']['classifier']
            classifier = make_instance(tgan2, conf_classifier)
            if 'model_path' in conf_classifier:
                chainer.serializers.load_npz(
                    conf_classifier['model_path'],
                    classifier, path=conf_classifier['npz_path'])
            if device >= 0:
                classifier = classifier.to_gpu()
            is_conf = config['inception_score']
            is_args = {
                'batchsize': is_conf['batchsize'],
                'n_samples': is_conf['n_samples'],
                'splits': is_conf['splits'],
                'n_frames': is_conf['n_frames'],
            }
            trainer.extend(
                tgan2.make_inception_score_extension(
                    gen, classifier, **is_args),
                trigger=(is_conf['interval'], 'iteration'))

        # Snapshot
        trainer.extend(
            extensions.snapshot_object(
                gen, 'generator_iter_{.updater.iteration}.npz'),
            trigger=snapshot_interval)
        # Do not save discriminator to save the space
        # trainer.extend(
        #     extensions.snapshot_object(
        #         dis, 'discriminator_iter_{.updater.iteration}.npz'),
        #     trigger=snapshot_interval)

        # Save movie
        if config.get('preview', None) is not None:
            preview_batchsize = config['preview']['batchsize']
            trainer.extend(
                tgan2.out_generated_movie(
                    gen, dis,
                    rows=config['preview']['rows'], cols=config['preview']['cols'],
                    seed=0, dst=config['out'], batchsize=preview_batchsize),
                trigger=snapshot_interval)

        # Log
        trainer.extend(extensions.LogReport(trigger=display_interval))
        report_keys = config['report_keys']
        if config.get('inception_score', None) is not None:
            report_keys.append('IS_mean')
        trainer.extend(extensions.PrintReport(report_keys), trigger=display_interval)
        trainer.extend(extensions.ProgressBar(update_interval=display_interval[0]))

    # Linear decay
    if ('linear_decay' in config) and (config['linear_decay']['start'] is not None):
        if comm.rank == 0:
            print('Use linear decay: {}:{} -> {}:{}'.format(
                config['linear_decay']['start'], config['iteration'],
                config['gen_opt']['args']['alpha'], 0.))
        trainer.extend(extensions.LinearShift(
            'alpha', (config['gen_opt']['args']['alpha'], 0.),
            (config['linear_decay']['start'], config['iteration']), gen_optimizer))
        trainer.extend(extensions.LinearShift(
            'alpha', (config['dis_opt']['args']['alpha'], 0.),
            (config['linear_decay']['start'], config['iteration']), dis_optimizer))

    # Checkpointer
    config_hash = hashlib.sha1()
    config_hash.update(yaml.dump(config_backup, default_flow_style=False).encode('utf-8'))
    os.makedirs('snapshots', exist_ok=True)
    checkpointer = chainermn.create_multi_node_checkpointer(
        name='tgan2', comm=comm, path=f'snapshots/{config_hash.hexdigest()}')
    checkpointer.maybe_load(trainer, gen_optimizer)
    if trainer.updater.epoch > 0:
        print('Resuming from checkpoints: epoch =', trainer.updater.epoch)
    trainer.extend(checkpointer, trigger=snapshot_interval)

    # Copy config to result dir
    os.makedirs(config['out'], exist_ok=True)
    config_path = os.path.join(config['out'], 'config.yml')
    with open(config_path, 'w') as fp:
        fp.write(yaml.dump(config_backup, default_flow_style=False))

    # Run the training
    trainer.run()


def parse_args():
    from tgan2.utils import make_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infiles', nargs='+', type=argparse.FileType('r'), default=())
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-c', '--comment', default='')
    parser.add_argument('-w', '--warning', action='store_true')
    parser.add_argument('-o', '--output-config', default='')
    args = parser.parse_args()

    conf_dicts = [yaml.load(fp) for fp in args.infiles]
    config = make_config(conf_dicts, args.attrs)
    return config, args


if __name__ == '__main__':
    config, args = parse_args()
    if not args.warning:
        # Ignore warnings
        warnings.simplefilter('ignore')
    if args.output_config != '':
        open(args.output_config, 'w').write(
            yaml.dump(config, default_flow_style=False))
    else:
        train(config)
