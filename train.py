from datetime import datetime
import argparse
import os
import numpy as np
import cv2
import torch
from torch import nn, optim
from torch.utils import data as torch_data
from tensorboardX import SummaryWriter
from coolname import generate_slug

from spair.models import Spair, ConvSpair
from spair import config as cfg
from spair.dataloader import SimpleScatteredMNISTDataset
from spair.manager import RunManager
from spair import debug_tools
from spair import metric


run_name = datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
run_log_path = 'logs_v2/%s' % run_name
writer = SummaryWriter(run_log_path)
print('log path:', run_log_path)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='Enable GPU use', action='store_true')
parser.add_argument('--no_z_prior', help='Enable GPU use', action='store_true')
parser.add_argument('--uniform_z_prior', help='Enable GPU use', action='store_true')
parser.add_argument('--conv_spair', help='Uses convolutional SPAIR rather than sequential SPAIR', action='store_true')
args = parser.parse_args()
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if args.no_z_prior:
    print("Z_PRES_PRIOR FLAG: no_z_prior")
elif args.uniform_z_prior:
    print('Z_PRES_PRIOR FLAG: uniform_z_prior')
else:
    print('Z_PRES_PRIOR FLAG: None')


def train():

    image_shape = cfg.INPUT_IMAGE_SHAPE
    dataset_name = 'spair/data/scattered_mnist_128x128_obj14x14.hdf5'
    print('dataset_name', dataset_name)

    data = SimpleScatteredMNISTDataset(dataset_name)
    run_manager = RunManager(run_name=run_name, dataset=data, device=device, writer=writer, run_args=args, cfg=cfg)

    torch.manual_seed(3)
    if args.conv_spair:
        spair_net = ConvSpair(image_shape).to(device)
        print('Running with CONV spair')
    else:
        spair_net = Spair(image_shape).to(device)
    params = spair_net.parameters()
    spair_optim = optim.Adam(params, lr=1e-4)

    # Main training loop
    for global_step, batch in run_manager.iterate_data():

        x_image, y_bbox, y_digit_count = batch

        x_image = x_image.to(device)
        y_bbox = y_bbox.to(device)
        y_digit_count = y_digit_count.to(device)


        print('Iteration', global_step)
        spair_optim.zero_grad()
        loss, out_img, z_where, z_pres = spair_net(x_image)
        print('\n ===> loss:', '{:.4f}'.format(loss.item()))
        loss.backward(retain_graph = True)
        spair_optim.step()

        # logging stuff
        image_out = out_img[0]
        image_in = x_image[0]
        combined_image = torch.cat([image_in, image_out], dim=2)
        writer.add_image('SPAIR input_output', combined_image,  global_step)

        # Log average precision metric every 5 step after 1000 iterations (when trainig_wheel is off)
        if global_step > 1000 and global_step % 5 == 0: # iteration > 1000 and
            meanAP = metric.mAP(z_where, z_pres, y_bbox, y_digit_count)
            print('Bbox Average Precision:', meanAP.item())
            writer.add_scalar('accuracy/bbox_average_precision', meanAP, global_step)

            count_accuracy = metric.object_count_accuracy(z_pres, y_digit_count)
            writer.add_scalar('accuracy/object_count_accuracy', count_accuracy, global_step)


        # Save model
        if global_step >= 1000 and global_step % 1000 == 0:
            check_point_name = 'step_%d.pkl' % global_step
            cp_dir = os.path.join(run_log_path, 'checkpoints')
            os.makedirs(cp_dir, exist_ok=True)
            save_path = os.path.join(run_log_path, 'checkpoints', check_point_name)
            torch.save(spair_net.state_dict(), save_path)
        # print('=================\n\n')
        torch.cuda.empty_cache()

    child_nr = 0



    for name, param in spair_net.named_children():
        print(name)




if __name__ == '__main__':
    train()