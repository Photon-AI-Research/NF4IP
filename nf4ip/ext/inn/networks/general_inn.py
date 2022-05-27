import torch.nn as nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
from nf4ip.ext.inn.handlers.inn_network_factory import InnNetworkFactoryHandler


class GeneralInn(InnNetworkFactoryHandler):
    class Meta:
        label = 'general_inn'

    def _get_structure(self, ndim_tot, num_blocks, feature):

        def subnet_fc(c_in, c_out):
            layers = [nn.Linear(c_in, feature), nn.LeakyReLU()]
            for i in range(0):
                layers.append(nn.Linear(feature, feature))
                layers.append(nn.LeakyReLU())

            layers.append(nn.Linear(feature, c_out))
            return nn.Sequential(*layers)

        nodes = [InputNode(ndim_tot, name='input')]

        for k in range(num_blocks):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':subnet_fc, 'clamp':2.0},
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed':k},
                              name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        return ReversibleGraphNet(nodes, verbose=False)