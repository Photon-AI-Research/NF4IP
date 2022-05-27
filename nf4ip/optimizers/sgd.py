import torch
from nf4ip.handlers.optimizer import OptimizerHandler

CONFIG = {
    'momentum': 0.9,
}


class SgdOptimizer(OptimizerHandler):
    class Meta:
        label = 'adam'

    def _get(self, params):
        return torch.optim.SGD(
            params,
            lr=self.app.config.get('nf4ip', 'lr'),
            momentum=self.app.config.get('sgd', 'momentum'),
            weight_decay=self.app.config.get('nf4ip', 'weight_decay')
        )
