import chainer
import chainer.functions as F

import numpy

try:
    import cupy
    cupy_enabled = True
except ImportError:
    cupy_enabled = False


class WGANGPUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        if not cupy_enabled:
            raise RuntimeError('DiracUpdater requires cupy')
        self.lam = kwargs.pop('lam')
        self.n_dis = kwargs.pop('n_dis')
        super(WGANGPUpdater, self).__init__(*args, **kwargs)

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
                x_array = arrays
                c = None
            else:
                x_array, c_array = arrays
                c = chainer.Variable(xp.asarray(c_array))
            x_real = chainer.Variable(xp.asarray(x_array))
            y_real = dis(x_real, c)

            batchsize = len(x_array)
            z = gen.make_hidden(batchsize)
            x_fake = gen(z, c)
            y_fake = dis(x_fake, c)

            if i == 0:
                loss_gen = F.sum(-y_fake) / batchsize
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.report({'loss_gen': loss_gen})
            x_fake.unchain_backward()

            eps = xp.asarray(numpy.random.uniform(0, 1, size=batchsize).astype('f'))
            eps = eps[:, None, None, None, None]  # N, n_frames, C, H, W
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            x_mid_v = chainer.Variable(x_mid.array)  # To cut the backward graph
            y_mid = dis(x_mid_v, c)
            dydx, = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)
            dydx = F.sqrt(F.sum(F.square(dydx), axis=(1, 2, 3, 4)))
            loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.array))

            loss_dis = F.sum(-y_real) / batchsize
            loss_dis += F.sum(y_fake) / batchsize

            dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.report({'loss_dis': loss_dis, 'loss_gp': loss_gp, 'g': F.mean(dydx)})
