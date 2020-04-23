import chainer
import chainer.functions as F
import chainer.links as L


class ConvLSTM(chainer.Chain):

    # Conv2D = EqualizedConv2D
    Conv2D = L.Convolution2D

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, dilate=1, peephole=False):
        super(ConvLSTM, self).__init__()
        with self.init_scope():
            self.w_xifoc = self.Conv2D(in_channels, out_channels * 4, ksize, stride, pad, dilate=dilate)
            self.w_hifoc = self.Conv2D(out_channels, out_channels * 4, ksize, stride, pad, dilate=dilate, nobias=True)

            if peephole:
                # Peephole
                initializer = chainer.initializers.Zero()
                self.peep_c_i = chainer.Parameter(initializer)
                self.peep_c_f = chainer.Parameter(initializer)
                self.peep_c_o = chainer.Parameter(initializer)

        self.out_channels = out_channels
        self.peephole = peephole
        self.c = None
        self.h = None

    def reset_state(self):
        self.c = None
        self.h = None

    def initialize_params(self, shape):
        self.peep_c_i.initialize((self.out_channels, shape[2], shape[3]))
        self.peep_c_f.initialize((self.out_channels, shape[2], shape[3]))
        self.peep_c_o.initialize((self.out_channels, shape[2], shape[3]))

    def initialize_state(self, shape):
        self.c = chainer.Variable(
            self.xp.zeros((shape[0], self.out_channels, shape[2], shape[3]), dtype=self.xp.float32))
        self.h = chainer.Variable(
            self.xp.zeros((shape[0], self.out_channels, shape[2], shape[3]), dtype=self.xp.float32))

    def __call__(self, x):
        # Initialize peephole weights
        if self.peephole and self.peep_c_i.array is None:
            self.initialize_params(x.shape)
        # Initialize state
        if self.c is None:
            self.initialize_state(x.shape)

        xifoc = self.w_xifoc(x)
        xi, xf, xo, xc = F.split_axis(xifoc, 4, axis=1)

        hifoc = self.w_hifoc(self.h)
        hi, hf, ho, hc = F.split_axis(hifoc, 4, axis=1)

        ci = F.sigmoid(xi + hi + (F.scale(self.c, self.peep_c_i, 1) if self.peephole else 0))
        cf = F.sigmoid(xf + hf + (F.scale(self.c, self.peep_c_f, 1) if self.peephole else 0))
        cc = cf * self.c + ci * F.tanh(xc + hc)
        co = F.sigmoid(xo + ho + (F.scale(cc, self.peep_c_o, 1) if self.peephole else 0))
        ch = co * F.tanh(cc)

        self.c = cc
        self.h = ch

        return ch
