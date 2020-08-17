from groupy.gconv.make_gconv_indices import make_c6_z2_indices, make_c6_p6_indices
from groupy.gconv.pytorch_gconv import splithexgconv2d


class P6ConvZ2(splithexgconv2d.SplitHexGConv2D):
    def __init__(self, *args, **kwargs):
        super(P6ConvZ2, self).__init__(*args, input_stabilizer_size=1, output_stabilizer_size=6, **kwargs)

    def make_transformation_indices(self):
        return make_c6_z2_indices(ksize=self.ksize)


class P6ConvP6(splithexgconv2d.SplitHexGConv2D):
    def __init__(self, *args, **kwargs):
        super(P6ConvP6, self).__init__(*args, input_stabilizer_size=6, output_stabilizer_size=6, **kwargs)

    def make_transformation_indices(self):
        return make_c6_p6_indices(ksize=self.ksize)
