from cement import App
from cement.core.foundation import TestApp
from cement.core.exc import CaughtSignal

from nf4ip.core.config import handle_config
from nf4ip.core.filtermanager import FilterManager
from nf4ip.controllers.tools import Tools
import horovod.torch as hvd

from nf4ip.interfaces.loss import LossInterface
from nf4ip.losses.mae import MaeLoss
from nf4ip.losses.mse import MseLoss
from nf4ip.losses.mmd_multiscale import MMDMultiscaleLoss

from nf4ip.interfaces.optimizer import OptimizerInterface
from nf4ip.optimizers.adam import AdamOptimizer, CONFIG as ADAM_CONFIG

from .core.exc import NF4IPError
from .controllers.base import Base
import torch

from nf4ip.core.abstactmodel import CONFIG



class NF4IP(App):
    """NF4IP primary application."""

    class Meta:
        label = 'nf4ip'

        # configuration defaults
        #config_defaults = CONFIG

        # call sys.exit() on close
        exit_on_close = True

        # load additional framework extensions
        extensions = [
            'generate',
            'yaml',
            'logging',
            'jinja2',
            'nf4ip.ext.inn'
        ]

        # configuration handler
        config_handler = 'yaml'

        # configuration file suffix
        config_file_suffix = '.yml'

        # set the log handler
        log_handler = 'logging'

        # set the output handler
        output_handler = 'jinja2'

        # register handlers
        handlers = [
            Base, Tools
        ]

        template_handler = 'jinja2'

    def __init__(self):
        super(NF4IP, self).__init__()
        hvd.init()
        self.device = torch.device("cuda:"+str(hvd.local_rank()) if torch.cuda.is_available() else "cpu")
        def load_config(app):
            config = getattr(self.pargs, 'nf4ip_config')
            if config is not None:
                self.config.parse_file(config)
        # parse the config parameter before the default configuration is processed
        self.hook.register('post_argument_parsing', load_config, -100)
        handle_config(self, 'nf4ip', CONFIG)
        self.hook.define('pre_training')
        self.hook.define('post_training')
        self.hook.define('pre_epoch')
        self.hook.define('post_epoch')
        self.hook.define('post_validate')

        self.extend('filter', FilterManager(self))
        self.filter.define('model_parameters')
        self.filter.define('checkpoint_save')
        self.hook.define('checkpoint_restore')
        self.filter.define('train_input')
        self.filter.define('val_input')
        self.filter.define('val_backward_output')
        self.filter.define('train_forward_output')
        self.filter.define('train_backward_output')
        self.filter.define('train_backward_rand_output')

        self.interface.define(LossInterface)
        self.handler.register(MaeLoss)
        self.handler.register(MseLoss)
        self.handler.register(MMDMultiscaleLoss)

        self.interface.define(OptimizerInterface)
        self.handler.register(AdamOptimizer)
        handle_config(self, 'adam', ADAM_CONFIG)


class NF4IPTest(TestApp, NF4IP):
    """A sub-class of nf4ip that is better suited for testing."""

    class Meta:
        label = 'nf4ip'


def main():
    with NF4IP() as app:
        try:
            app.run()

        except AssertionError as e:
            print('AssertionError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except NF4IPError as e:
            print('NF4IPError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except CaughtSignal as e:
            # Default Cement signals are SIGINT and SIGTERM, exit 0 (non-error)
            print('\n%s' % e)
            app.exit_code = 0


if __name__ == '__main__':
    main()
