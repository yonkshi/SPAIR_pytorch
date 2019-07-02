from datetime import datetime
import argparse

import numpy as np
import cv2
import torch
from torch import nn, optim
from torch.utils import data as torch_data
from tensorboardX import SummaryWriter
from coolname import generate_slug

from spair.models import SPAIR
from spair import config as cfg
from spair.dataloader import SimpleScatteredMNISTDataset
from spair import debug_tools

dt = datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
writer = SummaryWriter('logs_v2/%s' % dt)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='Enable GPU use', action='store_true')
args = parser.parse_args()
if args.gpu:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")


def train():

    image_shape = cfg.INPUT_IMAGE_SHAPE

    # Test image setup
    data = SimpleScatteredMNISTDataset('spair/data/scattered_mnist_14x14.hdf5')
    torch.manual_seed(3)

    spair_net = SPAIR(image_shape, writer, DEVICE).to(DEVICE)

    params = spair_net.parameters()
    spair_optim = optim.Adam(params, lr=1e-4)


    for epoch in range(100000):
        dataloader = torch_data.DataLoader(data,
                                            batch_size=cfg.BATCH_SIZE,
                                           pin_memory=True,
                                           num_workers= 1,
                                           )
        for batch_idx, batch in enumerate(dataloader):
            iteration = epoch * len(dataloader) + batch_idx
            batch = batch.to(DEVICE)
            print('Iteration', iteration)
            spair_optim.zero_grad()
            loss, out_img, z_where = spair_net(batch, iteration)
            loss.backward(retain_graph = True)
            spair_optim.step()

            # logging stuff
            image_out = out_img[0]
            image_in = batch[0]
            combined_image = torch.cat([image_in, image_out], dim=2)
            writer.add_image('SPAIR input_output', combined_image,  iteration)
            torch.cuda.empty_cache()
            print('=================\n\n')

    child_nr = 0



    for name, param in spair_net.named_children():
        print(name)




if __name__ == '__main__':
    train()