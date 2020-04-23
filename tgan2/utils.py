import functools
import warnings

import chainer
import chainer.links as L
import yaml

from tgan2.models.bn.categorical_conditional_batch_normalization \
    import CategoricalConditionalBatchNormalization

try:
    from chainermn.links import MultiNodeBatchNormalization
except Exception:
    warnings.warn('To perform batch normalization with multiple GPUs or '
                  'multiple nodes, MultiNodeBatchNormalization link is '
                  'needed. Please install ChainerMN: '
                  'pip install chainermn')


def make_instance(module, config, args=None):
    Class = getattr(module, config['name'])
    kwargs = config['args']
    if args is not None:
        kwargs.update(args)
    return Class(**kwargs)


def make_batch_normalization(channels, n_classes=0):
    if n_classes > 0:
        if (not hasattr(chainer.config, 'comm')) or chainer.config.comm is None:
            kwargs = {'n_cat': n_classes}
        else:
            kwargs = {'n_cat': n_classes, 'comm': chainer.config.comm}
        return CategoricalConditionalBatchNormalization(channels, **kwargs)
    else:
        if (not hasattr(chainer.config, 'comm')) or chainer.config.comm is None:
            return L.BatchNormalization(channels)
        else:
            return MultiNodeBatchNormalization(channels, comm=chainer.config.comm)


def make_config(conf_dicts, attr_lists=None):
    def merge_dictionary(base, diff):
        for key, value in diff.items():
            if (key in base and isinstance(base[key], dict)
                    and isinstance(diff[key], dict)):
                merge_dictionary(base[key], diff[key])
            else:
                base[key] = diff[key]

    config = {}
    for diff in conf_dicts:
        merge_dictionary(config, diff)
    if attr_lists is not None:
        for attr in attr_lists:
            module, new_value = attr.split('=')
            keys = module.split('.')
            target = functools.reduce(dict.__getitem__, keys[:-1], config)
            target[keys[-1]] = yaml.load(new_value)
    return config
