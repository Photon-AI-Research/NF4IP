from nf4ip.core.config import handle_config
import torch
from nf4ip.ext.vae.mlpvae import MLPVAE, Reversible, Config
from nf4ip.ext.vae.loss import vae_loss
CONFIG = {
    'enable': True,
    'load': None,
    'parallel_training': False,
}


class VaePlugin:
    def __init__(self, app):
        app.hook.register('post_argument_parsing', self.init_hook)
        handle_config(app, 'vae', CONFIG)
        self.app = app
        self.model = None
        self.parallel_training = False
        self.enable = True
        self.load_data = None

    def init_hook(self, app):
        self.enable = self.app.config.get('vae','enable')
        self.parallel_training = self.app.config.get('vae', 'parallel_training')
        if self.enable:
            self.app.hook.register('pre_training', self.init_vae)
            self.app.filter.register('train_input', self.processVae)
            self.app.filter.register('val_input', self.processVae)
            self.app.filter.register('checkpoint_save', self.checkpoint_save)
            self.app.filter.register('checkpoint_load', self.checkpoint_load)

            if self.parallel_training:
                # add the parameters only when autotraining
                self.app.filter.register('model_parameters', self.add_model_params)

    # the VAE gets lazy initialized on first use
    def init_vae(self, **kwargs):
        if self.model is not None:
            self.app.log.error("model is already initialized, preventing 2nd initialization!")
            return
        # VAE Network
        mlp_config = {
            'hidden_size': 100,
            'num_hidden': 8,
            'activation': torch.relu
        }
        config = Config(28, 1, 512)
        size = 34
        layers = 6
        lay = []
        for i in range(0, layers):
            lay.append(Reversible("conv", 3, ind=size, outd=size, pad=1))

        cnn_layers = [
            Reversible("conv", 3, ind=1, outd=size, input_layer=True),
            *lay,
            Reversible("pool", 3, ind=size, outd=size),
            Reversible("conv", 3, ind=size, outd=size),
            Reversible("pool", 3, ind=size, outd=size),
            Reversible("conv", 3, ind=size, outd=size),
            Reversible("pool", 3, ind=size, outd=size),
            Reversible("conv", 3, ind=size, outd=size)
        ]
        self.model = MLPVAE(config, mlp_config, cnn_layers).float().to(self.app.device)

        load_file = self.app.config.get('vae', 'load')
        if load_file is not None:
            self.app.log.info('loading VAE model from ' + load_file)
            self.model.load_state_dict(torch.load(load_file))
        elif self.load_data is not None:
            # load checkpoint data
            self.model.load_state_dict(self.load_data)
            self.load_data = None

    def checkpoint_save(self, checkpoint):
        checkpoint['vae_state_dict'] = self.model.state_dict()

    def checkpoint_load(self, checkpoint):
        self.load_data = checkpoint['vae_state_dict']

    def processVae(self, x, y, model, **kwargs):
        if self.model is None:
            # lazy initialization, just in case the pre_training hook was not called
            self.init_vae()
        in_net = x.unsqueeze(1)
        recon_x, z, mu, logvar = self.model.forward(in_net)
        if self.parallel_training:
            gamma = 10
            beta = 0.005194530884589891
            model.loss_forward += vae_loss(in_net, recon_x, mu, logvar, beta, gamma)

        return z, y

    def add_model_params(self, parameters, **kwargs):
        if self.model is None:
            # lazy initialization, just in case the pre_training hook was not called
            self.init_vae()
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        return parameters + trainable_parameters
