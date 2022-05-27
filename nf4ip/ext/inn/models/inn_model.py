import torch
import torch.optim

from nf4ip.core.abstactmodel import AbstractModel

# these are the config defaults that can be overridden by the project config or by command line arguments.
CONFIG = {
    'enable': True, #currently no effect
    'lambd_predict': 3.,
    'lambd_latent': 300.,
    'lambd_rev': 400.,
    'ndim_pad': 6,
    'ndim_z': 2,
    'inn_network_factory': 'general_inn',
    'loss_backward': 'mdd_multiscale',
    'loss_latent': 'mdd_multiscale',
    'loss_fit': 'mse',
}

# Based on "Analyzing Inverse Problems with Invertible Neural Networks" by L. Ardizzone et al.
class InnModel(AbstractModel):
    def __init__(self, app, device, train_loader, val_loader):
        super().__init__(app, device, train_loader, val_loader)

        self.ndim_pad = app.config.get('inn', 'ndim_pad')
        self.ndim_z = app.config.get('inn', 'ndim_z')

        # relative weighting of losses:
        self.lambd_predict = app.config.get('inn', 'lambd_predict')
        self.lambd_latent = app.config.get('inn', 'lambd_latent')
        self.lambd_rev = app.config.get('inn', 'lambd_rev')

        # MMD losses
        self.loss_func_backward = self.app.handler.get('loss', self.app.config.get('inn', 'loss_backward'), setup=True)
        self.loss_func_latent = self.app.handler.get('loss', self.app.config.get('inn', 'loss_latent'), setup=True)

        # Supervised loss
        self.loss_func_fit = self.app.handler.get('loss', self.app.config.get('inn', 'loss_fit'), setup=True)

        # will be initialized on training or loading
        self.ndim_total = 0
        self.ndim_y = 0
        self.ndim_x = 0

        self.loss_forward = None
        self.loss_backward = None

    def init_model(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        inn_model_factory = self.app.handler.get('inn_network_factory', self.app.config.get('inn', 'inn_network_factory'), setup=True)
        self.model = inn_model_factory.get_structure(self.ndim_total, self.num_blocks, self.feature).to(self.device)

        # restore model from checkpoint if available
        if self.model_state is not None:
            self.model.load_state_dict(self.model_state)
            self.model_state = None

        self.init_optimizer()
        self.model.train()


    def train_epoch(self, train_loader, n_epochs, i_epoch):
        if self.model is not None:
            self.model.train()

        l_tot = 0
        batch_idx = 0
        l_y = None
        l_z = None
        l_x_1 = None
        l_x_2 = None

        if self.loss_exp_scaling:
            # If MMD on x-space is present from the start, the self.model can get stuck.
            # Instead, ramp it up exponetially.
            self.loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / n_epochs)))
        for x, y in train_loader:
            self.loss_forward = torch.zeros(1, device=self.device)
            self.loss_backward = torch.zeros(1, device=self.device)
            # ndim_tot = ndim_y + ndim_z
            batch_idx += 1
            if self.max_batches_per_epoch is not None and batch_idx > self.max_batches_per_epoch:
                break
            x, y = x.float().to(self.device), y.float().to(self.device)
            x, y = self.app.filter.run('train_input', x, y, model=self)
            # initialize ndim_y and ndim_x on the actual size of the inputs
            if self.ndim_y == 0:
                self.ndim_y = y.size()[1]
            if self.ndim_x == 0:
                self.ndim_x = x.size()[1]

            self.ndim_total = self.ndim_y + self.ndim_z + self.ndim_pad

            if self.model is None:
                self.init_model()

            y_clean = y.clone()
            pad_x = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_total -
                                                         self.ndim_x, device=self.device)
            x = torch.cat((x, pad_x), dim=1)

            pad_yz = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_pad, device=self.device)
            y += self.y_noise_scale * torch.randn(self.batch_size, self.ndim_y, dtype=torch.float, device=self.device)
            y = torch.cat((torch.randn(self.batch_size, self.ndim_z, device=self.device), pad_yz, y), dim=1)

            self.optimizer.zero_grad()

            # Forward step:
            output = self.model(x)
            output = self.app.filter.run('train_forward_output', output, model=self)
            # Shorten output, and remove gradients wrt y, for latent loss
            y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

            l = self.lambd_predict * self.loss_func_fit.loss(output[:, self.ndim_z:], y[:, self.ndim_z:])
            l_y = l.data.item()

            output_block_grad = torch.cat((output[:, :self.ndim_z],
                                           output[:, -self.ndim_y:].data), dim=1)

            l_latent = self.lambd_latent * self.loss_func_latent.loss(output_block_grad, y_short)
            l += l_latent
            l_tot += l.data.item()
            l_z = l_latent.data.item()

            self.loss_forward += l
            self.loss_forward.backward(retain_graph=self.retain_graph)

            # Backward step:
            y = y_clean + self.y_noise_scale * torch.randn(self.batch_size, self.ndim_y, device=self.device)

            orig_z_perturbed = (output.data[:, :self.ndim_z] + self.y_noise_scale *
                                torch.randn(self.batch_size, self.ndim_z, device=self.device))
            rev_pad_yz = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_pad, device=self.device)
            y_rev = torch.cat((orig_z_perturbed, rev_pad_yz, y), dim=1)
            y_rev_rand = torch.cat((torch.randn(self.batch_size, self.ndim_z, device=self.device), rev_pad_yz, y), dim=1)

            output_rev = self.app.filter.run('train_backward_output', self.model(y_rev, rev=True), model=self)
            output_rev_rand = self.app.filter.run('train_backward_rand_output', self.model(y_rev_rand, rev=True), model=self)

            l_rev = (
                self.lambd_rev
                * self.loss_factor
                * self.loss_func_backward.loss(output_rev_rand[:, :self.ndim_x],
                                x[:, :self.ndim_x])
            )
            l_x_1 = l_rev.data.item()

            l_rev_2 = self.lambd_predict * self.loss_func_fit.loss(output_rev, x)
            l_rev += l_rev_2
            l_x_2 = l_rev_2.data.item()

            l_tot += l_rev.data.item()
            self.loss_backward += l_rev
            self.loss_backward.backward()
            for p in self.model.parameters():
                p.grad.data.clamp_(-15.00, 15.00)

            self.optimizer.step()
        if batch_idx == 0:
            raise RuntimeError("no batches were processed")
        # TODO: l_tot fürs gesammte batch und l_y - l_x_2 nur für mini batch prüfen
        return l_tot / batch_idx, {'l_y': l_y, 'l_z': l_z, 'l_x_1': l_x_1, 'l_x_2': l_x_2}

    # called to validate the model with validation data
    def validate(self, val_loader=None):
        if val_loader is None:
            val_loader = self.val_loader

        y_samps = None
        x_samps = None
        batch_idx = 0
        # collect the whole validaten dataset into one collection.
        for x, y in val_loader:
            batch_idx += 1
            if self.max_batches_per_validation is not None and batch_idx > self.max_batches_per_validation:
                break
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            x,y = self.app.filter.run('val_input', x, y, model=self)
            if(x_samps is None):
                x_samps = x
            else:
                x_samps = torch.cat((x_samps, x))
            if (y_samps is None):
                y_samps = y
            else:
                y_samps = torch.cat((y_samps, y))

        # the validation loss is computed on the [y,z] -> x direction
        y_samps_val = y_samps
        y_samps_val = torch.cat([torch.randn(x_samps.shape[0], self.ndim_z, device=self.device),
                                self.zeros_noise_scale * torch.zeros(x_samps.shape[0],
                                self.ndim_pad, device=self.device),
                                y_samps_val], dim=1)
        y_samps_val = y_samps_val.float()

        rev_x = self.model(y_samps_val, rev=True)
        rev_x = self.app.filter.run('val_backward_output', rev_x, model=self)
        predicted = rev_x[:, 0]

        ground_truth = x_samps

        loss = (torch.nn.MSELoss()(predicted, ground_truth[:, 0])).data

        for res in self.app.hook.run(
                'post_validate',
                model=self,
                i_epoch=self.i_epoch,
                n_epochs=self.n_epochs,
                loss=loss,
                x_samps=x_samps,
                y_samps=y_samps
        ):
            pass

        return loss
