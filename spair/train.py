import torch
from torch import nn, optim
import numpy as np
import cv2

from spair.models import SPAIR
from spair import config as cfg

def train():

    image_shape = cfg.INPUT_IMAGE_SHAPE

    # Test image setup
    img_BGR = cv2.imread('spair/data/testimg.png')
    img = img_BGR[..., ::-1].astype(np.float) # BGR to RGB
    img /= 255. # color space [0, 1]

    img = torch.from_numpy(np.array([np.moveaxis(img, [0,1,2], [1,2,0])], dtype=np.float32))
    imgs = img.repeat(1, 1, 1, 1)
    torch.manual_seed(5)

    spair_net = SPAIR(image_shape)

    params = spair_net.parameters()
    spair_optim = optim.Adam(params, lr=1e-4)



    for i in range(100):
        spair_optim.zero_grad()
        print('> begin learning', i)
        loss, out_img = spair_net(imgs)
        print('> begin computing loss')
        loss.backward(retain_graph=True)
        print('> loss done, begin computing spair', loss)
        spair_optim.step()
        print('> optim done')
        print('=================\n\n')

    child_nr = 0



    for name, param in spair_net.named_children():
        print(name)




if __name__ == '__main__':
    train()