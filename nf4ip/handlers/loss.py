from ..interfaces.loss import LossInterface
from cement import Handler


class LossHandler(LossInterface, Handler):

    def loss(self, input, target):
        return self._loss(input, target)