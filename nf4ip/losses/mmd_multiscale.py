import torch
from ..handlers.loss import LossHandler

class MMDMultiscaleLoss(LossHandler):
    class Meta:
        label = 'mmd_multiscale'

    def _loss(self, x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz
        # create the new tensors on the same device as x lives on.
        XX, YY, XY = (torch.zeros(xx.shape, device=x.device),
                      torch.zeros(xx.shape, device=x.device),
                      torch.zeros(xx.shape, device=x.device))

        for a in [0.05, 0.2, 0.9]:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

        return torch.mean(XX + YY - 2. * XY)
