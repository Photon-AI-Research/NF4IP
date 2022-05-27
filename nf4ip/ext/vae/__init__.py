from .vae_plugin import VaePlugin

def setup(app):
    app.vae = VaePlugin(app)

def load(app):
    app.hook.register('post_setup', setup)