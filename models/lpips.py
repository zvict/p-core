"""Thin LPIPS wrapper over the `lpips` package used for training and metrics."""

import torch.nn as nn
import lpips


class LPIPS(nn.Module):
    def __init__(self, net='vgg', reduce='mean'):
        super(LPIPS, self).__init__()
        self.reduce = reduce
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, x, y):
        if self.reduce == 'mean':
            return self.lpips(x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)).mean()
        else:
            raise NotImplementedError("Unknown reduce type in LPIPS: {}".format(self.reduce))
