
from pytest import raises
from nf4ip.main import NF4IPTest

def test_nf4ip():
    # test nf4ip without any subcommands or arguments
    with NF4IPTest() as app:
        app.run()
        assert app.exit_code == 0


def test_nf4ip_debug():
    # test that debug mode is functional
    argv = ['--debug']
    with NF4IPTest(argv=argv) as app:
        app.run()
        assert app.debug is True


def test_command1():
    # test command1 without arguments
    argv = ['command1']
    with NF4IPTest(argv=argv) as app:
        app.run()
        data,output = app.last_rendered
        assert data['foo'] == 'bar'
        assert output.find('Foo => bar')


    # test command1 with arguments
    argv = ['command1', '--foo', 'not-bar']
    with NF4IPTest(argv=argv) as app:
        app.run()
        data,output = app.last_rendered
        assert data['foo'] == 'not-bar'
        assert output.find('Foo => not-bar')
