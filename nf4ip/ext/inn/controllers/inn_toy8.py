from cement import Controller, ex

from nf4ip.ext.inn.datasets.toy8_dataset import Toy8Dataset

from ..models.inn_model import InnModel
from torch import torch
import matplotlib.pyplot as plt
import numpy as np


INN_CONFIG = {
    'ndim_pad': 6,
    'ndim_z': 2,
    'lambd_predict': 3.,
    'lambd_latent': 300.,
    'lambd_rev': 400.,
}

NF4IP_CONFIG = {
    'n_epochs': 50,
    'lr': 5e-4,
    'batch_size': 1600,
    'num_blocks': 8,
    'y_noise_scale': 1e-1,
    'zeros_noise_scale': 5e-2,
    'retain_graph': False,
    'max_batches_per_epoch': 8,
    'max_batches_per_validation': None,
    'loss_exp_scaling': True,
    'random_seed': 1987,
}

class INN(Controller):

    class Meta:
        label = 'inn'
        stacked_on = 'base'
        stacked_type = 'nested'

    def _default(self):
        self._parser.print_help()

    @ex(
        help='Train the toy8 network',
    )
    def toy8(self):
        """Train the network"""

        self.app.log.info("starting training")
        # this is a hack to get the defaults loaded without having a config file.
        # in your own application, use a yml config file instead.
        self.app.config.merge({'nf4ip': NF4IP_CONFIG}, override=True)
        self.app.config.merge({'inn': INN_CONFIG}, override=True)

        def printer(i_epoch, loss, **kwargs):
            print(i_epoch, loss)
        self.app.hook.register('post_epoch', printer)

        test_loader = torch.utils.data.DataLoader(
            Toy8Dataset(test=True),
            batch_size=self.app.config.get('nf4ip', 'batch_size'), shuffle=True, drop_last=True)

        train_loader = torch.utils.data.DataLoader(
            Toy8Dataset(test=False),
            batch_size=self.app.config.get('nf4ip', 'batch_size'), shuffle=True, drop_last=True)

        model = InnModel(self.app, self.app.device, train_loader, test_loader)

        model.train(train_loader)
        showResult(self.app.device, model, test_loader)


def showResult(device, model,  test_loader):
    N_samp = 4096

    x_samps = torch.cat([x for x, y in test_loader], dim=0)[:N_samp]
    y_samps = torch.cat([y for x, y in test_loader], dim=0)[:N_samp]
    c = np.where(y_samps)[1]
    y_samps += model.y_noise_scale * torch.randn(N_samp, model.ndim_y)
    y_samps = torch.cat([torch.randn(N_samp, model.ndim_z),
                         model.zeros_noise_scale * torch.zeros(N_samp, model.ndim_total - model.ndim_y - model.ndim_z),
                         y_samps], dim=1)
    y_samps = y_samps.to(device)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title('Predicted labels (Forwards Process)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('Generated Samples (Backwards Process)')
    # fig.show()
    # fig.canvas.draw()

    rev_x = model.model(y_samps, rev=True)
    rev_x = rev_x.cpu().data.numpy()

    pred_c = model.model(torch.cat((x_samps, torch.zeros(N_samp, model.ndim_total - model.ndim_x)),
                                  dim=1).to(device)).data[:, -8:].argmax(dim=1)
    axes[0].clear()
    axes[0].scatter(x_samps.cpu()[:, 0], x_samps.cpu()[:, 1], c=pred_c.cpu(), cmap='Set1', s=1., vmin=0, vmax=9)
    axes[0].axis('equal')
    axes[0].axis([-3, 3, -3, 3])
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].clear()
    axes[1].scatter(rev_x[:, 0], rev_x[:, 1], c=c, cmap='Set1', s=1., vmin=0, vmax=9)
    axes[1].axis('equal')
    axes[1].axis([-3, 3, -3, 3])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.canvas.draw()
    plt.show()

