
import h5py
import numpy as np

import torch
from torch.utils import data


class SimpleScatteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super().__init__()
        self.dataset = h5py.File(in_file, 'r')['test']
        self.episode = None

    def __getitem__(self, index):
        ret = []

        obs = self.dataset[0, ...] # TODO index fixed to 0
        obs = obs[..., None]  # Add channel dimension
        ret = np.moveaxis(obs, -1, 0)  # move from (x, y, c) to (c, x, y)

        return ret

    def __len__(self):
        return self.dataset.shape[0]

