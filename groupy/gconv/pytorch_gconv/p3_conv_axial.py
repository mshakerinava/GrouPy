from groupy.gconv.make_gconv_indices import make_c3_z2_indices, make_c3_p3_indices
from groupy.gconv.pytorch_gconv import splithexgconv2d


class P3ConvZ2(splithexgconv2d.SplitHexGConv2D):
    def __init__(self, *args, **kwargs):
        super(P3ConvZ2, self).__init__(*args, input_stabilizer_size=1, output_stabilizer_size=3, **kwargs)

    def make_transformation_indices(self):
        return make_c3_z2_indices(ksize=self.ksize)


class P3ConvP3(splithexgconv2d.SplitHexGConv2D):
    def __init__(self, *args, **kwargs):
        super(P3ConvP3, self).__init__(*args, input_stabilizer_size=3, output_stabilizer_size=3, **kwargs)

    def make_transformation_indices(self):
        return make_c3_p3_indices(ksize=self.ksize)
