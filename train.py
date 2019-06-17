from datetime import datetime
import argparse

import numpy as np
import cv2
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from coolname import generate_slug

from spair.models import SPAIR
from spair import config as cfg

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
    img_BGR = cv2.imread('spair/data/testimg.png')
    img = img_BGR[..., ::-1].astype(np.float) # BGR to RGB
    img /= 255. # color space [0, 1]

    img = torch.from_numpy(np.array([np.moveaxis(img, [0,1,2], [1,2,0])], dtype=np.float32))
    imgs = img.repeat(32, 1, 1, 1).to(DEVICE)
    torch.manual_seed(3)

    spair_net = SPAIR(image_shape, writer, DEVICE).to(DEVICE)

    params = spair_net.parameters()
    spair_optim = optim.Adam(params, lr=1e-4)


    for global_step in range(1000):
        print('Iteration', global_step)
        spair_optim.zero_grad()
        loss, out_img = spair_net(imgs, global_step)
        loss.backward(retain_graph = True)
        spair_optim.step()
        torch.cuda.empty_cache()
        print('=================\n\n')

    child_nr = 0



    for name, param in spair_net.named_children():
        print(name)




if __name__ == '__main__':
    train()