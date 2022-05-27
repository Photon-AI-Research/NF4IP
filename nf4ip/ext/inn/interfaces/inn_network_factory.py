from abc import abstractmethod
from cement import Interface


class InnNetworkFactoryInterface(Interface):
    class Meta:
        interface = 'inn_network_factory'

    @abstractmethod
    def _get_structure(self, ndim_tot, num_blocks, feature):
        """
        Get a network structure

        Returns:
            the model for the network
        """
        pass

    @abstractmethod
    def get_structure(self, ndim_tot, num_blocks, feature):
        """
        Get a network structure

        Returns:
            the model for the network
        """
        pass