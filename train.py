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
from spair import metric


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
    data = SimpleScatteredMNISTDataset('spair/data/scattered_mnist_128x128_obj14x14.hdf5')
    torch.manual_seed(3)

    spair_net = SPAIR(image_shape, writer, DEVICE).to(DEVICE)

    params = spair_net.parameters()
    spair_optim = optim.Adam(params, lr=1e-4)


    for epoch in range(100000):
        dataloader = torch_data.DataLoader(data,
                                            batch_size=cfg.BATCH_SIZE,
                                           pin_memory=True,
                                           num_workers= 1,
                                           drop_last = True,
                                           )
        for batch_idx, batch in enumerate(dataloader):
            x_image, y_bbox, y_digit_count = batch
            iteration = epoch * len(dataloader) + batch_idx

            x_image = x_image.to(DEVICE)
            y_bbox = y_bbox.to(DEVICE)
            y_digit_count = y_digit_count.to(DEVICE)


            print('Iteration', iteration)
            spair_optim.zero_grad()
            loss, out_img, z_where, z_pres = spair_net(x_image, iteration)
            loss.backward(retain_graph = True)
            spair_optim.step()

            # logging stuff
            image_out = out_img[0]
            image_in = x_image[0]
            combined_image = torch.cat([image_in, image_out], dim=2)
            writer.add_image('SPAIR input_output', combined_image,  iteration)

            # Log average precision metric every 5 step after 1000 iterations (when trainig_wheel is off)
            if iteration > 1000 and iteration % 5 == 0: # iteration > 1000 and
                meanAP = metric.mAP(z_where, z_pres,  y_bbox, y_digit_count)
                print('Bbox Average Precision:', meanAP.item())
                writer.add_scalar('accuracy/bbox_average_precision', meanAP, iteration)

            print('=================\n\n')
            torch.cuda.empty_cache()

    child_nr = 0



    for name, param in spair_net.named_children():
        print(name)




if __name__ == '__main__':
    train()