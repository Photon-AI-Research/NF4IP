import torch
from ..handlers.loss import LossHandler


class MseLoss(LossHandler):
    class Meta:
        label = 'mse'

    def _loss(self, input, target):
        return torch.nn.MSELoss()(input, target)
        #return torch.mean((input - target) ** 2) # yielts different results?
