
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from cycler import cycler
from matplotlib.collections import PatchCollection
import torch
import numpy as np
import time
import requests


from spair import config as cfg
from spair.manager import RunManager
from spair.logging import *
GRID_SIZE = 11
def plot_torch_image_in_pyplot( out:torch.Tensor, inp:torch.Tensor = None, batch_n=0):
    ''' For visualizing torch images in matplotlib '''
    torch_img = out[batch_n, ...]
    np_img = torch_img.detach().numpy()
    np_img = np.moveaxis(np_img,[0,1,2], [2,0,1]) # [C, H, W] -> [H, W, C]
    np_img = np.squeeze(np_img)
    plt.imshow(np_img)
    plt.title('out_image')
    plt.show()

    if inp is not None:
        torch_img = inp[batch_n, ...]
        np_img = torch_img.detach().numpy()
        np_img = np.moveaxis(np_img,[0,1,2], [2,0,1]) # [C, H, W] -> [H, W, C]
        plt.imshow(np_img)
        plt.show()

def benchmark_init():
    global BENCHMARK_INIT_TIME
    BENCHMARK_INIT_TIME = time.time()

def benchmark(name='', print_benchmark=True):
    global BENCHMARK_INIT_TIME
    now = time.time()
    diff = now - BENCHMARK_INIT_TIME
    BENCHMARK_INIT_TIME = now
    if print_benchmark: print('{} time: {:.4f} seconds'.format(name, diff))
    return diff

def torch2npy(t:torch.Tensor, reshape=False):
    '''
    Converts a torch graph node tensor (cuda or cpu) to numpy array
    :param t:
    :return:
    '''
    shape = t.shape[1:]
    if reshape:
        return t.cpu().view(cfg.BATCH_SIZE, GRID_SIZE, GRID_SIZE, *shape).detach().squeeze().numpy()
    return t.cpu().detach().numpy()

def plot_prerender_components(obj_vec, z_pres, z_depth, bounding_box, input_image):
    step = RunManager.global_step
    writer = RunManager.writer
    if step % 50 != 0:
        return

    ''' Plots each component prior to rendering '''
    # obj_vec = obj_vec.view(32, 11, 11, 28, 28, 3)
    obj_vec = torch2npy(obj_vec, reshape=True)
    obj_vec = obj_vec[0, ...]
    obj_vec = np.concatenate(obj_vec, axis=-3) # concat h
    obj_vec = np.concatenate(obj_vec, axis=-2) # concat w
    # z_pres = z_pres.view(32, 11, 11, 1)
    if RunManager.run_args.z_pres == 'self_attention':
        batch_size, hw, _ = z_pres.shape
        unit = torch.ones([batch_size, 1, hw]).to(RunManager.device)
        z_pres = torch.bmm(unit, z_pres).view(batch_size*hw).contiguous()
    z_pres = torch2npy(z_pres, reshape=True)
    z_depth = torch2npy(z_depth, reshape=True)
    bounding_box = torch2npy(bounding_box, reshape=True)
    input_image = input_image[0,...].permute(1,2,0).cpu().detach().squeeze().numpy()

    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize = (10,7))
    fig.tight_layout()
    plt.tight_layout()


    # Attr image
    obj = obj_vec[...,0]
    _plot_image('rendered_obj', obj, gs[0, 0], fig)

    # Alpha Channel (heatmap)
    alpha = obj_vec[...,1]
    _plot_heatmap('alpha', alpha, gs[0, 1], fig, cmap='spring')

    # Importance (heatmap)
    impo = obj_vec[...,2]
    _plot_heatmap('importance', impo, gs[0, 2], fig, cmap='summer')

    # Bounding Box
    bbox = bounding_box[0, ...] * cfg.INPUT_IMAGE_SHAPE[-2] # image size
    presence = z_pres[0, ...]
    _plot_bounding_boxes('bounding boxes', bbox, input_image, presence,  gs[1,0], fig)

    # depth (heatmap)
    depth = z_depth[0,...]
    _plot_heatmap('z_depth', depth, gs[1, 1], fig, cmap='autumn')
    # Presence (heatmap)

    _plot_heatmap('z_presence', presence, gs[1, 2], fig, cmap='winter')

    if cfg.IS_LOCAL:
        plt.show()
        print('hello world')
    else:
        writer.add_figure('renderer_analysis', fig, step)

def plot_cropped_input_images(cropped_input_images):
    step = RunManager.global_step
    writer = RunManager.writer

    input_imgs = cropped_input_images.permute(0,4,5, 2,3,1,).cpu().squeeze().detach().numpy()
    # np.swapaxes(input_imgs, )
    input_img = input_imgs[0,...]
    H = input_img.shape[0]
    W = input_img.shape[1]

    # adding border to cropped images
    px_h = px_w = input_img.shape[-1] + 2
    input_img_with_border = np.ones([H, W, px_h, px_w])
    input_img_with_border[..., 1:-1, 1:-1] = input_img


    img = np.concatenate(input_img_with_border, axis=-2) # concat h
    img = np.concatenate(img, axis=-1) # concat w

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, cmap='gray' , vmin=0, vmax=1)

    if cfg.IS_LOCAL:
        plt.show()
        print('hello world')
    else:
        writer.add_figure('debug_cropped_input_images', fig, step)

def plot_objet_attr_latent_representation(z_attr, title='z_attr/heatmap'):

    step = RunManager.global_step
    writer = RunManager.writer

    z_attr = z_attr[0, ...]
    z_attr = torch2npy(z_attr)

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize = (7,2.5))
    fig.tight_layout()
    plt.tight_layout()

    z_attr_max = z_attr.max(axis=0)
    _plot_heatmap('Max', z_attr_max, gs[0, 0], fig, cmap='spring')

    z_attr_mean = z_attr.mean(axis=0)
    _plot_heatmap('Mean', z_attr_mean, gs[0, 1], fig, cmap='spring')

    z_attr_min = z_attr.min(axis=0)
    _plot_heatmap('Min', z_attr_min, gs[0, 2], fig, cmap='spring')

    if cfg.IS_LOCAL:
        plt.show()
        print('hello world')
    else:
        writer.add_figure(title, fig, step)


def _plot_heatmap(title, data, gridspec, fig, cmap, vmin=0, vmax=1):
    ax = fig.add_subplot(gridspec)
    im = ax.imshow(data, cmap=cmap)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

def _plot_image(title, data, gridspec, fig):
    ax = fig.add_subplot(gridspec)
    im = ax.imshow(data, cmap='gray' , vmin=0, vmax=1) # Specific to the MNIST dataset
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

def _plot_bounding_boxes(title, bbox, original_image, z_pres, gridspec, fig):

    ax = fig.add_subplot(gridspec)
    ax.imshow(original_image, cmap='gray', vmin=0, vmax=1)
    #ptchs = []
    H, W, _ = bbox.shape
    for i in range(H):
        for j in range(W):
            x, y, w, h = bbox[i,j]
            pres = np.clip(z_pres[i, j], 0.2, 1)
            border_color = (1,0,0,pres) if pres > 0.5 else (0,0,1, pres)# red box if > 0.5, otherwise blue

            # Green box: ground truth, red box: inferrence, blue box: disabled inferrence

            x -= w/2
            y -= h/2
            patch = patches.Rectangle([x,y], w, h, facecolor='none', edgecolor=border_color, linewidth=1)
            ax.add_patch(patch)

    # ax.add_collection(PatchCollection(ptchs, facecolors='none', edgecolors='r', linewidths=1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def decoder_output_grad_hook(grad):
    step = RunManager.global_step
    writer = RunManager.writer

    if torch.isnan(grad).sum() == 0:
        return
    obj_px = cfg.OBJECT_SHAPE[0]
    grad = grad.view(cfg.BATCH_SIZE, GRID_SIZE, GRID_SIZE, obj_px, obj_px, 2)
    nan_locations = torch.isnan(grad).nonzero()

    print('nan_locations', nan_locations)

    obj_px = cfg.OBJECT_SHAPE[0]
    grad = grad.view(cfg.BATCH_SIZE, GRID_SIZE, GRID_SIZE, obj_px, obj_px, 2).cpu().detach().numpy()
    grad = grad[0, ...]
    obj_vec = np.concatenate(grad, axis=-3) # concat h
    obj_vec = np.concatenate(obj_vec, axis=-2) # concat w
    img = obj_vec[...,0]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, vmin=-1e-4, vmax=1e-4)
    plt.title('gradient of decoder')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    if cfg.IS_LOCAL:
        plt.show()
        print('')
    else:
        writer.add_figure('grad_visualization/decoder_out', fig, step)

def grad_nan_hook(name, grad):
    sum = torch.isnan(grad).sum()
    if torch.isnan(grad).sum() == 0:
        return
    log('!! ===== NAN FOUND IN GRAD ====')
    log(name)
    log('shape',grad.shape,', total nans:', sum )
    log('location', torch.isnan(grad).nonzero())
    log('===============================')

    telegram_yonk('We found a nan in grad')



def z_attr_grad_hook(grad):
    step = RunManager.global_step
    writer = RunManager.writer

    if step % 50 != 0:
        return
    # grad = grad.view(2, cfg.N_ATTRIBUTES, 11, 11, 2).squeeze().detach().numpy()
    z_attr_grad = torch2npy(grad[0, ...])

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize = (7,2.5))
    fig.tight_layout()
    plt.tight_layout()

    z_attr_max = z_attr_grad.max(axis=0)
    _plot_heatmap('Max', z_attr_max, gs[0, 0], fig, cmap='spring')

    z_attr_mean = z_attr_grad.mean(axis=0)
    _plot_heatmap('Mean', z_attr_mean, gs[0, 1], fig, cmap='spring')

    z_attr_min = z_attr_grad.min(axis=0)
    _plot_heatmap('Min', z_attr_min, gs[0, 2], fig, cmap='spring')

    if cfg.IS_LOCAL:
        plt.show()
        print('')
    else:
        writer.add_figure('grad_visualization/z_attr', fig, step)

def nan_hunter(hunter_name, **kwargs):
    step = RunManager.global_step

    nan_detected = False
    tensors = {}
    non_tensors = {}
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            tensors[name] = value
            if torch.isnan(value).sum() > 0:
                nan_detected = True
        else:
            non_tensors[name] = value

    if not nan_detected: return


    log('======== NAN DETECTED in %s =======' % RunManager.run_name)
    log('Nan Hunter Name', hunter_name)
    log('global_step', step)
    for name, value in non_tensors.items():
        log(name, value)

    for name, tensor in tensors.items():
        tensor_size = tensor.nelement()
        tensor = torch.isnan(tensor).sum().item()
        log('{} NaN/total elements'.format(name), '{} / {}'.format(tensor, tensor_size))

    log('======== END OF NAN DETECTED =======')

    telegram_yonk('NaN Detected!! {}, step: {}'.format(RunManager.run_name, RunManager.global_step))

    raise AssertionError('NAN Detected by Nan detector')





