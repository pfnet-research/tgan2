import chainer
import chainer.functions as F

import numpy

try:
    import cupy
    cupy_enabled = True
except ImportError:
    cupy_enabled = False


def _expand(option):
    if isinstance(option, (list, tuple)):
        return list(option)
    elif option:
        return [2] * 64
    else:
        return [1] * 64


class DiracUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        if not cupy_enabled:
            raise RuntimeError('DiracUpdater requires cupy')
        self.lam = kwargs.pop('lam')
        self.n_dis = kwargs.pop('n_dis')
        self.reg_type = kwargs.pop('reg_type')
        self.subsample_batch = _expand(kwargs.pop('subsample_batch'))
        super(DiracUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('generator')
        dis_optimizer = self.get_optimizer('discriminator')
        gen = gen_optimizer.target
        dis = dis_optimizer.target
        xp = gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            arrays = chainer.dataset.concat_examples(batch)
            if isinstance(arrays, (numpy.ndarray, cupy.ndarray)):
                x_real = chainer.Variable(xp.asarray(arrays))
                c = None
                batchsize = len(x_real)
            else:

                def make_x_real(arr):
                    x_real = []
                    subsample_batch = [1] + self.subsample_batch
                    stride = numpy.cumprod(subsample_batch[:len(arr)])
                    for a, s in zip(arr, stride):
                        a = a[::s]
                        x_real.append(chainer.Variable(xp.asarray(a)))
                    return tuple(x_real)

                if arrays[-1].dtype == numpy.int32:
                    x_real = make_x_real(arrays[:-1])
                    c = chainer.Variable(xp.asarray(arrays[-1]))
                else:
                    x_real = make_x_real(arrays)
                    c = None
                batchsize = len(x_real[0])
            y_real = dis(x_real, c)

            z = gen.make_hidden(batchsize)
            x_fake = gen(z, c)
            y_fake = dis(x_fake, c)

            if i == 0:
                loss_gen = F.mean(F.softplus(-y_fake))
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.report({'loss_gen': loss_gen})

            if isinstance(x_fake, tuple):
                for xf in x_fake:
                    xf.unchain_backward()
            else:
                x_fake.unchain_backward()

            if self.reg_type == 'fake':
                xd, yd = x_fake, y_fake
            elif self.reg_type == 'real':
                xd, yd = x_real, y_real
            else:
                raise TypeError
            xd = list(xd) if isinstance(xd, tuple) else [xd]
            dydxs = chainer.grad([yd], xd, enable_double_backprop=True)

            loss_gp = None
            for dydx in dydxs:
                loss = F.sum(F.square(dydx))
                if loss_gp is None:
                    loss_gp = loss
                else:
                    loss_gp += loss
            loss_gp = self.lam * loss_gp

            loss_dis = F.mean(F.softplus(-y_real)) + F.mean(F.softplus(y_fake))

            dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()
            dis.cleargrads()

            chainer.report({'loss_dis': loss_dis, 'loss_gp': loss_gp, 'g': F.mean(dydx)})
