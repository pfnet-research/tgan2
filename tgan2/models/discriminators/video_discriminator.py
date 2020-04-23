import chainer
import chainer.functions as F
import chainer.links as L


class VideoDiscriminator(chainer.Chain):

    def __init__(self, in_channels, top_width, mid_ch, n_classes=0):
        super(VideoDiscriminator, self).__init__()
        w = None
        with self.init_scope():
            self.c0 = L.ConvolutionND(3, in_channels, mid_ch, 4, 2, 1, initialW=w)
            self.c1 = L.ConvolutionND(3, mid_ch, mid_ch * 2, 4, 2, 1, initialW=w)
            self.c2 = L.ConvolutionND(3, mid_ch * 2, mid_ch * 4, 4, 2, 1, initialW=w)
            self.c3 = L.ConvolutionND(3, mid_ch * 4, mid_ch * 8, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(mid_ch * 8, 1, top_width, 1, 0, initialW=w)
            if n_classes > 0:
                self.l_c = L.EmbedID(n_classes, mid_ch * 8 * top_width * top_width, initialW=w)

    def __call__(self, x, c=None):
        self.x = x
        h0 = F.leaky_relu(self.c0(x))
        h1 = F.leaky_relu(self.c1(h0))
        h2 = F.leaky_relu(self.c2(h1))
        h3 = F.leaky_relu(self.c3(h2))
        h4 = F.reshape(h3, (h3.shape[0] * h3.shape[2],) + self.c4.W.shape[1:])
        output = self.c4(h4).reshape(len(h4), 1)
        if c is not None:
            h = h4.reshape(len(h4), -1)
            w_c = self.l_c(c)
            output += F.sum(w_c * h, axis=1, keepdims=True)
        return output
