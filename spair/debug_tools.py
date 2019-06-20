
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from cycler import cycler
from matplotlib.collections import PatchCollection
import torch
import numpy as np
import time

from spair import config as cfg

def plot_torch_image_in_pyplot( out:torch.Tensor, inp:torch.Tensor = None, batch_n=0):
    ''' For visualizing '''
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

def benchmark(name=''):
    global BENCHMARK_INIT_TIME
    now = time.time()
    diff = now - BENCHMARK_INIT_TIME
    BENCHMARK_INIT_TIME = now
    print('{}: {:.4f} '.format(name, diff))

def torch2npy(t:torch.Tensor):
    '''
    Converts a torch graph node tensor (cuda or cpu) to numpy array
    :param t:
    :return:
    '''
    shape = t.shape[1:]
    # TODO Change batch size back
    return t.cpu().view(32, 11, 11, *shape).detach().squeeze().numpy()

def plot_prerender_components(obj_vec, z_pres, z_depth, bounding_box, writer, step):
    ''' Plots each component prior to rendering '''
    # obj_vec = obj_vec.view(32, 11, 11, 28, 28, 3)
    obj_vec = torch2npy(obj_vec)
    obj_vec = obj_vec[0, ...]
    obj_vec = np.concatenate(obj_vec, axis=-3) # concat h
    obj_vec = np.concatenate(obj_vec, axis=-2) # concat w
    # z_pres = z_pres.view(32, 11, 11, 1)
    z_pres = torch2npy(z_pres)
    z_depth = torch2npy(z_depth)
    bounding_box = torch2npy(bounding_box)

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
    bbox = bounding_box[0, ...] * 128 # image size
    _plot_bounding_boxes('bounding boxes', bbox, gs[1,0], fig)

    # depth (heatmap)
    depth = z_depth[0,...]
    _plot_heatmap('z_depth', depth, gs[1, 1], fig, cmap='autumn')
    # Presence (heatmap)
    presence = z_pres[0,...]
    _plot_heatmap('z_presence', presence, gs[1, 2], fig, cmap='winter')

    if cfg.IS_LOCAL:
        plt.show()
        print('hello world')
    else:
        writer.add_figure('renderer_analysis', fig, step)

def plot_debug_rendered_output(rendered, writer, step):

    pass


def _plot_heatmap(title, data, gridspec, fig, cmap):
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
    ax.set_title(title)

def _plot_bounding_boxes(title, bbox, gridspec, fig):
    ax = fig.add_subplot(gridspec)
    bg = np.zeros([128, 128])
    ax.imshow(bg)
    #ptchs = []
    for rows in bbox:
        for cols in rows:
            x, y, w, h = cols
            patch = patches.Rectangle([x,y], w, h, facecolor='none', edgecolor='r', linewidth=1)
            #ptchs.append(patch)
            ax.add_patch(patch)

    # ax.add_collection(PatchCollection(ptchs, facecolors='none', edgecolors='r', linewidths=1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

