
from cement import Controller, ex
from cement.utils.version import get_version_banner
from ..core.version import get_version

VERSION_BANNER = """
{{ description }} %s
%s
""" % (get_version(), get_version_banner())


class Base(Controller):
    class Meta:
        label = 'base'

        # text displayed at the top of --help output
        description = '{{ description }}'

        # text displayed at the bottom of --help output
        epilog = 'Usage: {{ label }} train'

        # controller level arguments. ex: '{{ label }} --version'
        arguments = [
            ### add a version banner
            ( [ '-v', '--version' ],
              { 'action'  : 'version',
                'version' : VERSION_BANNER } ),
        ]


    def _default(self):
        """Default action if no sub-command is passed."""

        self.app.args.print_help()


    @ex(
        help='example training command',
    )
    def train(self):
        """train the network"""

        self.app.log.info("example training command!")
        my_option = self.app.config.get('{{ label }}','my_option')
        self.app.log.info("my_option is {}".format(my_option))
