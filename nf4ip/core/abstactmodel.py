import os
import pathlib
from abc import abstractmethod
from datetime import datetime
import horovod.torch as hvd
from cement import Interface
import torch
from tqdm import tqdm

CONFIG = {
    # application params
    'data_dir': './data',
    'run': None,
    '_run': str,
    'config': None,
    '_config': str,

    # learning params
    'n_epochs': 50,
    'lr': 1e-4,
    'weight_decay': 2e-5,
    'batch_size': 1600,
    'feature': 512,
    'num_blocks': 8,
    'y_noise_scale': 1e-1,
    'zeros_noise_scale': 5e-2,
    'optimizer': 'adam',

    #misc
    'retain_graph': False,
    'max_batches_per_epoch': None,
    'max_batches_per_validation': None,
    '_max_batches_per_validation': int,
    'loss_exp_scaling': False,
    'random_seed': None,
    '_random_seed': int,
    'validate_every_epochs': None,
    '_validate_every_epochs': int,
    'checkpoint_every_epochs': 5,
    'overwrite': False,
}


class AbstractModel(Interface):
    class Meta:
        interface = 'nf4ipmodel'

    def __init__(self, app, device, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.app = app
        self.device = device
        # Training parameters
        self.batch_size = app.config.get('nf4ip', 'batch_size')

        self.lr = app.config.get('nf4ip', 'lr')
        self.feature = app.config.get('nf4ip', 'feature')
        self.num_blocks = app.config.get('nf4ip', 'num_blocks')

        self.retain_graph = app.config.get('nf4ip', 'retain_graph')
        self.max_batches_per_epoch = app.config.get('nf4ip', 'max_batches_per_epoch')
        self.max_batches_per_validation = app.config.get('nf4ip', 'max_batches_per_validation')
        self.loss_exp_scaling = app.config.get('nf4ip', 'loss_exp_scaling')
        self.random_seed = app.config.get('nf4ip', 'random_seed')
        self.loss_factor = 1
        self.n_epochs = self.app.config.get('nf4ip', 'n_epochs')

        self.y_noise_scale = app.config.get('nf4ip', 'y_noise_scale')
        self.zeros_noise_scale = app.config.get('nf4ip', 'zeros_noise_scale')

        self.validate_every_epochs = self.app.config.get('nf4ip', 'validate_every_epochs')
        self.checkpoint_every_epochs = self.app.config.get('nf4ip', 'checkpoint_every_epochs')
        self.overwrite = self.app.config.get('nf4ip', 'overwrite')
        self.run = self.app.config.get('nf4ip', 'run')
        if self.run is not None:
            self.app.log.info("### Run " + self.run + " ###")

        # model and optimizer are lazy initialized on training
        self.model = None
        self.optimizer = None
        self.i_epoch = 1

        # the state is loaded early at a time the model/optimizer is not initialized
        # these variables store the restored state until the model is ready.
        # do not forget to set them back to None after the states are restored to save memory.
        self.model_state = None
        self.optimizer_state = None

        if self.run is not None:
            self.checkpoint_restore()

    def get_trainable_parameters(self):
        # Set up the optimizer
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        # allow other plugins to add parameters (like VAE)
        trainable_parameters = self.app.filter.run('model_parameters', trainable_parameters, model=self)

        if self.optimizer_state is None:
            # only initialize parameters when the model is not loaded
            for param in trainable_parameters:
                param.data = 0.05 * torch.randn_like(param)

        return trainable_parameters

    def init_optimizer(self):
        parameters = self.get_trainable_parameters()
        optimizer = self.app.handler.get('optimizer', self.app.config.get('nf4ip', 'optimizer'), setup=True)\
            .get(parameters)

        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None

        self.optimizer = hvd.DistributedOptimizer(optimizer,
             named_parameters=self.model.named_parameters(),
             #backward_passes_per_step=args.batches_per_allreduce,
             #op=hvd.Adasum if args.use_adasum else hvd.Average
            )

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    @abstractmethod
    def train_epoch(self, train_loader, n_epochs, i_epoch):
        pass

    @abstractmethod
    def validate(self):
        pass

    def train(self, train_loader=None):
        if train_loader is None:
            train_loader = self.train_loader

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        if self.i_epoch == self.n_epochs:
            self.app.log.error("already reached the last epoch (%d) on this checkpoint, nothing to do here." %
                               self.i_epoch)
            exit(1)

        for res in self.app.hook.run(
                'pre_training',
                model=self,
        ):
            pass

        for self.i_epoch in tqdm(range(self.i_epoch, self.n_epochs + 1)):
            self.app.log.debug("Starting epoch {}".format(self.i_epoch))
            for res in self.app.hook.run(
                    'pre_epoch',
                    model=self,
                    i_epoch=self.i_epoch,
                    n_epochs=self.n_epochs,
            ):
                pass

            loss, info_dict = self.train_epoch(train_loader, self.n_epochs, self.i_epoch)

            for res in self.app.hook.run(
                    'post_epoch',
                    model=self,
                    i_epoch=self.i_epoch,
                    n_epochs=self.n_epochs,
                    loss=loss,
                    info_dict=info_dict
            ):
                pass

            if self.validate_every_epochs is not None \
                    and self.i_epoch % self.validate_every_epochs == 0 \
                    and self.i_epoch != 1 or self.i_epoch == self.n_epochs:
                self.validate()

            if self.run is not None \
                    and self.checkpoint_every_epochs is not None \
                    and self.i_epoch % self.checkpoint_every_epochs == 0 \
                    and self.i_epoch != 1 or self.i_epoch == self.n_epochs:
                self.checkpoint_save()

        for res in self.app.hook.run(
                'post_training',
                model=self,
        ):
            pass

    def checkpoint_save(self, file_path=None):
        if hvd.rank() != 0:
            return

        checkpoint = {
            'config': self.app.config.get_dict(),
            'i_epoch': self.i_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        self.app.log.debug("Saving checkpoint at epoch {}".format(self.i_epoch))

        checkpoint = self.app.filter.run('checkpoint_save', checkpoint)
        torch.save(checkpoint, self._get_checkpoint_file(file_path))

    def checkpoint_restore(self, file_path=None):
        checkpoint_file = self._get_checkpoint_file(file_path)
        # do not restore when overwrite flag is set
        if os.path.isfile(checkpoint_file) and not self.overwrite:
            checkpoint = torch.load(checkpoint_file)
            self.app.config.merge(checkpoint['config'], override=True)
            self.i_epoch = checkpoint['i_epoch']
            self.app.log.info("restoring from checkpoint {} at epoch {}".format(checkpoint_file, self.i_epoch))
            self.model_state = checkpoint['model_state_dict']
            self.optimizer_state = checkpoint['optimizer_state_dict']

            for res in self.app.hook.run('checkpoint_restore', checkpoint):
                pass

    def _get_checkpoint_file(self, file_path):
        if file_path is not None:
            checkpoint_path = file_path
            pathlib.Path(checkpoint_path).parents[1].mkdir(parents=True, exist_ok=True)
            return file_path

        base_path = self.app.config.get('nf4ip', 'data_dir')
        checkpoint_path = os.path.join(base_path, 'checkpoints')
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        if self.run is None:
            # use temporary name
            run = "unnamed"
        else:
            run = self.run

        return os.path.join(checkpoint_path, run)
