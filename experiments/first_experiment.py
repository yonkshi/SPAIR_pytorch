
from datetime import datetime
import torch
import numpy as np
from spair import config as cfg
from tensorboardX import SummaryWriter
from coolname import generate_slug

from spair.dataloader import SimpleScatteredMNISTDataset
from spair.manager import RunManager
from train import train
from spair.metric import *

# TODO DONE Check if max is selecting the correct bbox
# TODO DONE Check with a smaller average [0: 0.5: 0.9] for example, and verify average
# TODO DONE Check if the label works correctly with -1
# TODO Check if xy are alligned between label and bounding box
np.random.seed(1337)
batch_size = 32
grid_size = 11
num_label = 5
z_where = np.zeros((batch_size,grid_size, grid_size,4))
grid_space = np.linspace(0, 1.0, 11, dtype=np.float)[..., None]
z_where[..., 1] += grid_space
z_where[..., 0] += grid_space.T
z_where[...,2:] = 0.15

z_where = z_where.reshape((batch_size, grid_size * grid_size, 4))
z_pres = np.ones((batch_size,grid_size * grid_size,1))

ground_truth_bbox = np.zeros((batch_size, num_label, 4))
offset = np.random.choice(grid_space.squeeze(), (num_label, 2) )
ground_truth_bbox[...,:2] += offset
ground_truth_bbox[...,2:] = 0.1

ground_truth_bbox[:, -2:,:] = -1
truth_bbox_digit_count = np.ones((batch_size, 1)) * (num_label - 2)


# conversion stuff
z_where_t = torch.tensor(z_where, dtype=torch.float32)
z_where_t = z_where_t.view(batch_size,grid_size, grid_size, -1).permute(0,3,1,2)
z_pres_t = torch.tensor(z_pres, dtype=torch.float32)
z_pres_t = z_pres_t.view(batch_size,grid_size, grid_size, -1).permute(0,3,1,2)
ground_truth_bbox_t = torch.tensor(ground_truth_bbox, dtype=torch.float32) * 128
truth_bbox_digit_count_t = torch.tensor(truth_bbox_digit_count, dtype=torch.float32)

result = mAP(z_where_t, z_pres_t, ground_truth_bbox_t, truth_bbox_digit_count_t)
print(result.item())

