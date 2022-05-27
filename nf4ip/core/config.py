
# this function accepts a module config with default values,
# merges them into the current config object and sets up command line arguments.
# The type of the argument is determined by the default value.
# if the default is "None" and your config key is mykey, you can specify
# '_mykey': type
# to set the type within the config dictionary.
# generally, keys starting with _ are ignored.
def handle_config(app, module, config):
    filtered_config = {k: v for k, v in config.items() if not k.startswith('_')}

    def pre_argument_parsing(_app):
        # merge current config
        app.config.merge({module: filtered_config}, override=False)
        if module == 'nf4ip':
            # nf4ip is default module on cli, so it does not need to be specified
            climodule = ''
        else:
            climodule = module + '.'

        # setup command line arguments
        for (key, val) in filtered_config.items():
            # check if there is a _-version of the key that sets the type, if yes use it.
            if '_' + key in config:
                argtype = config.get('_'+key)
            else:
                argtype = type(val)

            clikey = key.replace('_', '-')
            if argtype == bool:
                # for boolean arguments, add switches with "--arg" and "--no-arg" that set true or false
                # instead of "--arg False"
                _app.args.add_argument('--' + climodule + clikey, action='store_true', dest=module + '_' + key)
                _app.args.add_argument('--no-' + climodule + clikey, action='store_false', dest=module + '_' + key)
                _app.args.set_defaults(**{module + '_' + key: None})
            else:
                _app.args.add_argument('--' + climodule + clikey, action='store', dest=module + '_' + key, type=argtype)

    def post_argument_parsing(_app):
        # read back parsed command line arguments and override config
        for (key, val) in filtered_config.items():
            if getattr(_app.pargs, module + '_' + key) is not None:
                _app.config.set(module, key, getattr(_app.pargs, module + '_' + key))
        config1 = _app.config.get_dict()
        # convert string "None" to None type
        for key, val in config1[module].items():
            if type(val) == str and val == 'None':
                _app.config.set(module, key, None)

    app.hook.register('pre_argument_parsing', pre_argument_parsing)
    app.hook.register('post_argument_parsing', post_argument_parsing, -50)