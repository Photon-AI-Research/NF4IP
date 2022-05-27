import os

from cement import minimal_logger
from nf4ip.core.config import handle_config
from tensorboardX import SummaryWriter

CONFIG = {
    'enable': True,
    'logdir': './data/tensorboard',
    'log_metrics_every_epochs': 1
}


class TensorBoard:

    def __init__(self, app):
        self.app = app

        app.hook.register('post_argument_parsing', self.init_hook)
        handle_config(app, 'tensorboard', CONFIG)
        self.log_writer = None
        self.enable = None


    def init_hook(self, app):
        self.enable = self.app.config.get('progressbar', 'enable')
        if not self.enable:
            return
        self.logdir = self.app.config.get('tensorboard', 'logdir')
        self.log_metrics_every_epochs = self.app.config.get('tensorboard', 'log_metrics_every_epochs')
        self.run = self.app.config.get('nf4ip', 'run')
        if self.run is None:
            self.run = "unnamed"

        self.log_writer = SummaryWriter(os.path.join(self.logdir, self.run))
        self.app.hook.register('post_epoch', self.epoch_log_hook)
        self.app.hook.register('post_validate', self.validate_log_hook)

    def epoch_log_hook(self,
                       model,
                       i_epoch,
                       n_epochs,
                       loss,
                       info_dict,
                       **kwargs):
        # LOG.debug('Writing epoch statistics', __name__)
        if self.log_metrics_every_epochs is not None \
                and i_epoch % self.log_metrics_every_epochs == 0 \
                and i_epoch != 1 or i_epoch == n_epochs:
            self.log_writer.add_scalar('Loss', loss, i_epoch)
            for key, val in info_dict.items():
                self.log_writer.add_scalar(key, val, i_epoch)

            self.log_writer.flush()

    # hook that loggs the validation loss
    def validate_log_hook(self, i_epoch, loss, **kwargs):
        self.log_writer.add_scalar('validation_loss', loss.item(), i_epoch)


def load(app):
    app.tensorboard = TensorBoard(app)
