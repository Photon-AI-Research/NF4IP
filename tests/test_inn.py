import pytest
from nf4ip.main import NF4IPTest
import numpy as np
import torch
import torch.utils.data
from nf4ip.ext.inn.models.inn_model import InnModel
from functools import reduce

def generate(labels, tot_dataset_size):
    # print('Generating artifical data for setup "%s"' % (labels))
    verts = [
        (-2.4142, 1.),
        (-1., 2.4142),
        (1., 2.4142),
        (2.4142, 1.),
        (2.4142, -1.),
        (1., -2.4142),
        (-1., -2.4142),
        (-2.4142, -1.)
    ]

    label_maps = {
        'all': [0, 1, 2, 3, 4, 5, 6, 7],
        'some': [0, 0, 0, 0, 1, 1, 2, 3],
        'none': [0, 0, 0, 0, 0, 0, 0, 0],
    }

    np.random.seed(0)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.2)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.

    shuffling = np.random.permutation(N)
    pos = torch.tensor(pos[shuffling], dtype=torch.float)
    labels = torch.tensor(labels[shuffling], dtype=torch.float)

    return pos, labels


# this test (based on the toy8 example) trains a model for 2 epochs
# computes the sum of the model for each epoch end compares it to predefined (expected and correct) values.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_nf4ip_inn_model():
    return True
    with NF4IPTest() as app:
        sums = []
        def hook(model, **kwargs):
            a = model.model.state_dict()
            sum = reduce((lambda x, y: x.sum() + y.sum()), a.values())
            sums.append(round(sum.item(),2))
        app.hook.register('post_epoch', hook)

        torch.manual_seed(1234)

        batch_size = 1600
        test_split = 10000
        
        pos, labels = generate(
            labels='all',
            tot_dataset_size=2 ** 20
        )

        ndim_pad = 6
        ndim_z = 2
        feature = 512
        num_blocks = 8
        lr = 1e-3
        # our test object does not have pre-set configuration parameters for ease of testing
        # therefore we need to set every individual parameter ourself
        app.config.set('nf4ip', 'batch_size', batch_size)
        app.config.set('nf4ip', 'lr', lr)
        app.config.set('nf4ip', 'feature', feature)
        app.config.set('nf4ip', 'num_blocks', num_blocks)
        app.config.set('nf4ip', 'retain_graph', True)
        app.config.set('nf4ip', 'max_batches_per_epoch', 8)
        app.config.set('nf4ip', 'max_batches_per_validation', 8)
        app.config.set('nf4ip', 'loss_exp_scaling', True)
        app.config.set('nf4ip', 'retain_graph', True)
        app.config.set('nf4ip', 'random_seed', 2342)
        app.config.set('nf4ip', 'n_epochs', 2)
        app.config.set('nf4ip', 'y_noise_scale', 1e-1)
        app.config.set('nf4ip', 'zeros_noise_scale', 5e-2)
        app.config.set('nf4ip', 'optimizer', 'adam')
        app.config.set('nf4ip', 'validate_every_epochs', None)
        app.config.set('nf4ip', 'checkpoint_every_epochs', 5)
        app.config.set('nf4ip', 'overwrite', True)
        app.config.set('nf4ip', 'run', 'test')
        app.config.set('nf4ip', 'data_dir', 'data/')
        
        app.config.add_section('inn')
        app.config.set('inn', 'ndim_pad', ndim_pad)
        app.config.set('inn', 'ndim_z', ndim_z)
        app.config.set('inn', 'feature', feature)
        app.config.set('inn', 'num_blocks', num_blocks)
        
        app.config.set('inn', 'lambd_predict', 3.)
        app.config.set('inn', 'lambd_latent', 300.)
        app.config.set('inn', 'lambd_rev', 400.)
        
        app.config.set('inn', 'loss_backward', 'mmd_multiscale')
        app.config.set('inn', 'loss_latent', 'mmd_multiscale')
        app.config.set('inn', 'loss_fit', 'mse')
        app.config.set('inn', 'inn_network_factory', 'general_inn')
        
        app.config.add_section('adam')
        app.config.set('adam', 'betas', [0.8, 0.9])
        app.config.set('adam', 'amsgrad', False)
        app.config.set('adam', 'eps', 1e-6)
        app.config.set('nf4ip', 'weight_decay', 2e-5)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model = InnModel(app, device, ndim_pad, ndim_z, feature, num_blocks, batch_size, lr, lambd_predict=3.,
        #                 lambd_latent=300., lambd_rev=400.)

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos[:test_split], labels[:test_split]),
            batch_size=batch_size, shuffle=True, drop_last=True)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
            batch_size=batch_size, shuffle=True, drop_last=True)
        
        model = InnModel(app, device, train_loader, test_loader)
        model.train()
        assert sums == [1939.63, 1998.69]
        #assert sums == [37080.13, 3875.88]
