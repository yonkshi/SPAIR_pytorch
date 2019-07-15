from datetime import datetime
import logging
import sys
import argparse
import os
import time
import requests

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
from spair.logging import *


def main():
    run_name = datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
    log_path = 'logs/%s' % run_name

    args = parse_args(log_path)
    # Setup TensorboardX writer
    writer = SummaryWriter(args.log_path)
    # Setup logger
    init_logger(log_path)

    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    dataset_path = 'data/' + args.dataset_filename
    dataset_subset_name = args.dataset_subset
    data = SimpleScatteredMNISTDataset(dataset_path, dataset_subset_name)

    run_manager = RunManager(run_name=run_name, dataset=data, device=device, writer=writer, run_args=args)

    if not cfg.IS_LOCAL:
        try:
            train(run_manager)
        except Exception as e:
            telegram_yonk('An error had occured:{}, step:{}'.format(run_name, RunManager.global_step) )
            raise e
    else:
        train(run_manager)

def train(run_manager):

    image_shape = cfg.INPUT_IMAGE_SHAPE
    device = run_manager.device
    writer = run_manager.writer
    run_log_path = run_manager.run_args.log_path

    torch.manual_seed(3)
    np.random.seed(3)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if run_manager.run_args.original_spair:
        spair_net = Spair(image_shape)
    else:
        spair_net = ConvSpair(image_shape)
        print('Running with CONV spair')


    if continue_training(run_manager.run_args) is not None:
        model_dict, global_step = continue_training(run_manager.run_args)
        spair_net.load_state_dict(model_dict)
        RunManager.global_step = global_step

    spair_net.to(device)
    params = spair_net.parameters()
    spair_optim = optim.Adam(params, lr=1e-4)
    benchmark_time = time.time()
    debug_tools.benchmark_init()
    # Main training loop
    for global_step, batch in run_manager.iterate_data():

        x_image, y_bbox, y_digit_count = batch
        x_image = x_image.to(device)
        y_bbox = y_bbox.to(device)
        y_digit_count = y_digit_count.to(device)

        log('Iteration', global_step)
        debug_tools.benchmark_init()
        spair_optim.zero_grad()
        loss, out_img, z_where, z_pres = spair_net(x_image)
        # log('===> loss:', '{:.4f}'.format(loss.item()))

        loss.backward(retain_graph = True)
        spair_optim.step()

        # Log average precision metric every 5 step after 1000 iterations (when trainig_wheel is off)
        if global_step > 1000 and global_step % 5 == 0: # iteration > 1000 and
            meanAP = metric.mAP(z_where, z_pres, y_bbox, y_digit_count)
            log('Bbox Average Precision:', meanAP.item())
            writer.add_scalar('accuracy/bbox_average_precision', meanAP, global_step)

            count_accuracy = metric.object_count_accuracy(z_pres, y_digit_count)
            writer.add_scalar('accuracy/object_count_accuracy', count_accuracy, global_step)

        # Save model
        if global_step % 100 == 0 and global_step > 0:
            # logging stuff
            image_out = out_img[0]
            image_in = x_image[0]
            combined_image = torch.cat([image_in, image_out], dim=2)
            writer.add_image('SPAIR input_output', combined_image, global_step)

            duration = time.time() - benchmark_time
            writer.add_scalar('misc/time_taken_per_100_batches', duration, global_step)
            check_point_name = 'step_%d.pkl' % global_step
            cp_dir = os.path.join(run_log_path, 'checkpoints')
            os.makedirs(cp_dir, exist_ok=True)
            save_path = os.path.join(run_log_path, 'checkpoints', check_point_name)
            torch.save(spair_net.state_dict(), save_path)
            benchmark_time = time.time()

        # print('=================\n\n')

        # torch.cuda.empty_cache()


    telegram_yonk('Run completed! name:{}'.format(run_manager.run_name))

    for name, param in spair_net.named_children():
        print(name)

def continue_training(args):
    if not args.continue_from:
        return

    log_path = args.continue_from
    checkpoint_num = args.check_point_number
    filename = 'step_{}.pkl'.format(checkpoint_num)
    model_pkl = os.path.join('logs/', log_path, 'checkpoints', filename)
    model_dict = torch.load(model_pkl)
    return model_dict, checkpoint_num


def parse_args(run_log_path):
    parser = argparse.ArgumentParser()
    # Run config
    parser.add_argument('--gpu', help='Enable GPU use', action='store_true')
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='max number of iterations to train on')

    # Core Algorithm config
    parser.add_argument('--z_pres', type=str, default='original_prior',
                        choices=['original_prior', 'no_prior', 'uniform_prior', 'self_attention'], help='name of the dataset')

    parser.add_argument('--original_spair', help='Uses sequential SPAIR rather than convolutional SPAIR',
                        action='store_true')

    parser.add_argument('--conv_neighbourhood', type=int, default=1,
                        help='kernel size of conv_spair')
    parser.add_argument('--use_z_where_decoder', help='Use a decoder to model z_where better', action='store_true')

    parser.add_argument('--use_uber_trick', help='Attaches explicit x,y information to input image for better localization', action='store_true')

    parser.add_argument('--use_conv_z_attr', help='Use a conv network to learn z_attr for faster learning', action='store_true')

    parser.add_argument('--hw_prior', type=float, default=[3., 0.5], nargs=2,
                        help='z prior for the height and width of bbox')

    parser.add_argument('--backbone_self_attention', help='Enable self attention modules for the backbone network', action='store_true')

    # Dataset config
    parser.add_argument('--dataset_subset', type=str, default='constant',
                        choices=['constant', 'full', '1-5'], help='name of the dataset')

    parser.add_argument('--dataset_filename', type=str, default='scattered_mnist_128x128_obj14x14.hdf5',
                        help='name of the dataset')

    # Logging config
    parser.add_argument('--log_path', type=str, default=run_log_path,
                        help='path of to store logging and checkpoints')

    # Checkpoint restoration
    parser.add_argument('--continue_from', type=str, default="",
                        help='name of run to continue trainig from')

    parser.add_argument('--check_point_number', type=int, default=0,
                        help='checkpoint number to load from')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    main()