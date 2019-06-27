
import h5py
import numpy as np
from spair import config as cfg
import torch
from torch.utils import data
import cv2


class SimpleScatteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super().__init__()
        self.dataset = h5py.File(in_file, 'r')['test']
        self.episode = None

        static_img = self.dataset[9, ...]
        img_size = cfg.INPUT_IMAGE_SHAPE[-1]
        self.static_img = cv2.resize(static_img, dsize=(img_size,img_size))

    def __getitem__(self, index):
        ret = []

        obs = self.dataset[9, ...] # TODO index fixed to 0
        # obs = self.static_img
        # obs = np.zeros_like(obs)
        obs = obs[..., None]  # Add channel dimension
        ret = np.moveaxis(obs, -1, 0)  # move from (x, y, c) to (c, x, y)

        return ret

    def __len__(self):
        return self.dataset.shape[0]

