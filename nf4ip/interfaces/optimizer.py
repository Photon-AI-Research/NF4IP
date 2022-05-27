from abc import abstractmethod
from cement import Interface


class OptimizerInterface(Interface):
    class Meta:
        interface = 'optimizer'

    @abstractmethod
    def _get(self, parameters):
        pass

    @abstractmethod
    def get(self, parameters):
        pass
