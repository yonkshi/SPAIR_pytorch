import torch
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
    imgs = img.repeat(32, 1, 1, 1)
    torch.manual_seed(5)

    spair_net = SPAIR(image_shape)

    out = spair_net(imgs)



if __name__ == '__main__':
    train()