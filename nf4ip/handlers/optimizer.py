from ..interfaces.optimizer import OptimizerInterface
from cement import Handler


class OptimizerHandler(OptimizerInterface, Handler):

    def get(self, parameters):
        return self._get(parameters)