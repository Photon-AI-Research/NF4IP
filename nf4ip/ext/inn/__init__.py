
import os
from .controllers.inn_toy8 import INN
from nf4ip.core.config import handle_config
from nf4ip.ext.inn.models.inn_model import CONFIG
from .interfaces.inn_network_factory import InnNetworkFactoryInterface
from .networks.general_inn import GeneralInn

def load(app):
    handle_config(app, 'inn', CONFIG)
    app.handler.register(INN)
    app.interface.define(InnNetworkFactoryInterface)
    app.handler.register(GeneralInn)




