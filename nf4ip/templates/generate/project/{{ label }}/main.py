
from cement import App, TestApp, init_defaults
from cement.core.exc import CaughtSignal
from nf4ip.main import NF4IP
import os
from .core.exc import {{ class_name }}Error
from .controllers.base import Base
from nf4ip.controllers.tools import Tools
from nf4ip.core.config import handle_config

# configuration defaults
CONFIG = {
    'my_option': 100
}

class {{ class_name }}(NF4IP):
    """{{ name }} primary application."""

    class Meta:
        label = '{{ label }}'

        # call sys.exit() on close
        exit_on_close = True

        # load additional framework extensions
        extensions = [
            'yaml',
            'logging',
            'jinja2',
            'nf4ip.ext.inn',
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

        project_config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')

    def __init__(self):
        super({{ class_name }}, self).__init__()
        handle_config(self, '{{ label }}', CONFIG)
        self.hook.register('post_setup', self.post_setup)


    def post_setup(self, app):
        self.config.parse_file(os.path.join(self.Meta.project_config_dir, self.Meta.label + '.yml'))


class {{ class_name }}Test(TestApp,{{ class_name }}):
    """A sub-class of {{ class_name }} that is better suited for testing."""

    class Meta:
        label = '{{ label }}'


def main():
    with {{ class_name }}() as app:
        try:
            app.run()

        except AssertionError as e:
            print('AssertionError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except {{ class_name }}Error as e:
            print('{{ class_name }}Error > %s' % e.args[0])
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
