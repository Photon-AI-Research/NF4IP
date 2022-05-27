from torch.utils.data import Dataset
import torch
import numpy as np
import math

class Toy8Dataset(Dataset):
    def __init__(self, test=False):
        test_split = 10000
        pos, labels = self.generate(
            labels='all',
            tot_dataset_size=2 ** 20
        )

        if test:
            self.pos = pos[:test_split]
            self.labels = labels[:test_split]
        else:
            self.pos = pos[test_split:]
            self.labels = labels[test_split:]

    def generate(self, labels, tot_dataset_size):
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
        n = N // 8

        for i, v in enumerate(verts):
            pos[i * n:(i + 1) * n, :] += v
            labels[i * n:(i + 1) * n, mapping[i]] = 1.

        shuffling = np.random.permutation(N)
        pos = torch.tensor(pos[shuffling], dtype=torch.float)
        labels = torch.tensor(labels[shuffling], dtype=torch.float)

        return pos, labels

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, index):
        pos = self.pos[index]
        label = self.labels[index]
        return pos, label
