import torch
from ..handlers.loss import LossHandler


class MaeLoss(LossHandler):
    class Meta:
        label = 'mae'

    def _loss(self, input, target):
        return torch.mean(abs(input - target))
