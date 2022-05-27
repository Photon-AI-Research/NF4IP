from nf4ip.core.config import handle_config
from tqdm import tqdm

# TODO: add a custom log handler that uses pbar.write() functionality for proper logging output
CONFIG = {
    'enable': True,
}

class Progressbar:
    def __init__(self, app):
        self.app = app
        self.pbar = None
        self.info = dict()
        self.run = None
        self.enable = True
        handle_config(app, 'progressbar', CONFIG)
        app.hook.register('post_argument_parsing', self.init_hook)

    def init_hook(self, app):
        self.enable = self.app.config.get('progressbar', 'enable')
        if self.enable:
            self.run = self.app.config.get('nf4ip', 'run')
            self.app.hook.register('pre_training', self.pre_training_hook, 50)
            self.app.hook.register('post_epoch', self.post_epoch_hook, 50)
            self.app.hook.register('post_validate', self.post_validate_hook, 50)

    def pre_training_hook(self, model, **kwargs):
        options = {
            'total': model.n_epochs,
            'ncols': 0,
            'postfix': self.info,
            'initial': model.i_epoch,
        }
        if self.run is not None:
            options['desc'] = self.run
        self.pbar = tqdm(**options)
        pass

    def post_epoch_hook(self, i_epoch, n_epochs, loss, **kwargs):

        self.info['loss'] = loss
        self.pbar.set_postfix(self.info)
        if i_epoch <= n_epochs:
            self.pbar.update()

    def post_validate_hook(self, loss, **kwargs):
        self.info['val_loss'] = loss.item()

    def write(self, msg):
        self.pbar.write(msg)


def load(app):
    app.progressbar = Progressbar(app)
