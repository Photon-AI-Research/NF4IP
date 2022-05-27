import torch
from nf4ip.handlers.optimizer import OptimizerHandler

CONFIG = {
    'eps': 1e-6,
    'betas': [0.8, 0.9],
    'amsgrad': False,
}

class AdamOptimizer(OptimizerHandler):
    class Meta:
        label = 'adam'

    def _get(self, params):
        return torch.optim.Adam(
            params,
            lr=self.app.config.get('nf4ip', 'lr'),
            betas=tuple(self.app.config.get('adam', 'betas')),
            eps=self.app.config.get('adam', 'eps'),
            weight_decay=self.app.config.get('nf4ip', 'weight_decay'),
            amsgrad=self.app.config.get('adam', 'amsgrad')
        )

