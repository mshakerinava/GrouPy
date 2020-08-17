import torch
from groupy import hexa
from groupy.gconv.pytorch_gconv import splitgconv2d


class SplitHexGConv2D(splitgconv2d.SplitGConv2D):
    """
    """
    def __init__(self, *args, image_shape='square', **kwargs):
        """
        """
        super(SplitHexGConv2D, self).__init__(*args, **kwargs)

        self.image_shape = image_shape
        filter_mask = hexa.mask.hexagon_axial(self.ksize)[None, None, None, ...]
        self.register_buffer('filter_mask', torch.tensor(filter_mask, dtype=torch.float32))

    def forward(self, x):
        # Apply a mask to the parameters
        with torch.no_grad():
            self.weight.data = self.weight.data * self.filter_mask

        y = super(SplitHexGConv2D, self).forward(x)

        # Get a square shaped mask if it does not yet exist.
        if not hasattr(self, 'output_mask'):
            ny, nx = y.data.shape[-2:]
            if self.image_shape == 'square':
                gen_output_mask = hexa.mask.square_axial
            elif self.image_shape == 'triangle':
                gen_output_mask = hexa.mask.triangle_axial
            else:
                assert False, '`image_shape` should be "square" or "triangle"'
            output_mask = gen_output_mask(ny, nx)[None, None, None, ...]
            self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.float32, device=y.device))

        y = y * self.output_mask

        return y
