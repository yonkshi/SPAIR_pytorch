
import h5py
import numpy as np
from spair import config as cfg
import torch
from torch.utils import data
import cv2


class SimpleScatteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super().__init__()
        self.dataset = h5py.File(in_file, 'r')['train/constant']
        self.episode = None

        # static_img = self.dataset[9, ...]
        # img_size = cfg.INPUT_IMAGE_SHAPE[-1]
        # self.static_img = cv2.resize(static_img, dsize=(img_size,img_size))

    def __getitem__(self, index):
        ret = []

        obs = self.dataset['image'][index, ...]
        # obs = self.static_img
        # obs = np.zeros_like(obs)
        obs = obs[..., None]  # Add channel dimension
        image = np.moveaxis(obs, -1, 0)  # move from (x, y, c) to (c, x, y)

        bbox = self.dataset['bbox'][index, ...]

        digit_count = self.dataset['digit_count'][index, ...]

        return image, bbox, digit_count

    def __len__(self):
        return self.dataset['image'].shape[0]

