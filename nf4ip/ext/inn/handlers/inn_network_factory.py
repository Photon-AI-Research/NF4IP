from ..interfaces.inn_network_factory import InnNetworkFactoryInterface
from cement import Handler


class InnNetworkFactoryHandler(InnNetworkFactoryInterface, Handler):

    def get_structure(self, ndim_tot, num_blocks, feature):
        return self._get_structure(ndim_tot, num_blocks, feature)