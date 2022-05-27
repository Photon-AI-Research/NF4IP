from abc import abstractmethod
from cement import Interface


class LossInterface(Interface):
    class Meta:
        interface = 'loss'

    @abstractmethod
    def _loss(self, input, target):
        """
        Calculate the loss

        Returns:
            loss value
        """
        pass

    @abstractmethod
    def loss(self, input, target):
        """
        Calculate the loss

        Returns:
            loss value
        """
        pass