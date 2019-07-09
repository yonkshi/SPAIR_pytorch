from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, Linear, Module, ModuleDict, ModuleList
from torch.nn import functional as F

from spair import config as cfg
from spair import debug_tools
from spair.manager import RunManager

class Backbone(Module):
    def __init__(self, input_shape, n_out_channels, topology=cfg.DEFAULT_BACKBONE_TOPOLOGY, internal_activation=ReLU):
        '''
        Builds the primary backbone network for feature extraction. CNN with no output activation function
        :param n_in_channels:
        :param n_out_channels:
        :param topology:
        :param internal_activation:
        :return:
        '''
        super().__init__()
        self.topology = topology
        n_in_channels = input_shape[0] # Assuming pytorch style [C, H, W] tensor, ignoring batch
        self.input_shape = input_shape

        self.net = self._build_backbone(n_in_channels, n_out_channels)

        self.padding, self.n_grid_cells, self.grid_cell_size = self._build_receptive_field_padding()


    def compute_output_shape(self):
        '''
        Computes the feature space output dimensions based on input image shape
        :return: shape of the feature space vector
        '''
        t = torch.rand(1, *cfg.INPUT_IMAGE_SHAPE)
        # out = self.forward(t)
        out = self.__call__(t)
        out_shape = out.shape[1:] # remove the batch
        return out_shape

    def _build_backbone(self, n_in_channels, n_out_channels):
        '''Builds the convnet of the backbone'''

        n_prev = n_in_channels
        net = OrderedDict()

        # Builds internal layers except for the last layer
        for i, layer in enumerate(self.topology):
            layer['in_channels'] = n_prev

            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f #rename
            else:
                f = layer['out_channels']

            net['conv_%d' % i] = Conv2d(**layer)
            net['act_%d' % i] = ReLU()
            n_prev = f

        # Builds the final layer
        net['conv_out'] = Conv2d(in_channels=f, out_channels=n_out_channels, kernel_size=1, stride=1)

        return Sequential(net)

    def _build_receptive_field_padding(self):
        '''Computes corresponding receptive field for each output pixel'''
        '''
        Pads the input tensor for backbone network so that the output can map to a specific region of the input image
        :param x:
        :return:
        '''

        j = np.array([1,1])
        r = np.array([1,1])
        receptive_fields = []

        for layer in self.topology:
            kernel_size = np.array(layer['kernel_size'])  # for each layer, [4, 4, 4, 1, .. ]
            stride = np.array(layer['stride'])  # for each layer, [3, 2, 2, 1, .. ]
            r = r + (kernel_size - 1) * j  # starts at [1, 1] + 3 * [1,1]
            j = j * stride  # cumulative ratio
            receptive_fields.append(dict(size=r, translation=j))

        # computes the output layer's ratio to input layer
        grid_cell_size = receptive_fields[-1]["translation"]

        # compute the output layer's absolute size
        rf_size = receptive_fields[-1]["size"]
        pre_padding = np.floor(rf_size / 2 - grid_cell_size / 2).astype('i')

        image_shape = self.input_shape[-2:] # Gets the H and W, assuming [N, C, H, W] style tensor
        n_grid_cells = np.ceil(image_shape / grid_cell_size).astype('i')
        required_image_size = rf_size + (n_grid_cells - 1) * grid_cell_size
        post_padding = required_image_size - image_shape - pre_padding

        pad_top = pre_padding[0] # Height padding
        pad_bottom = post_padding[0]
        pad_left = pre_padding[1]
        pad_right = post_padding[1]


        return nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom)), n_grid_cells, grid_cell_size

    def forward(self, x):
        padded_x = self.padding(x)
        # debug_tools.plot_stn_input_and_out(padded_x)
        out = self.net(padded_x)
        return out

class LatentConv(Module):
    '''
    Special Convolution Network for learning latent variables
    '''
    def __init__(self, in_channels, out_channels, additional_out_channels = None):
        super().__init__()
        neighbourhood = RunManager.run_args.conv_neighbourhood
        self.kernel_size = (neighbourhood - 1) * 2 + 1
        self.input_pad = nn.ZeroPad2d(neighbourhood - 1)
        self.out_split_size = None

        if additional_out_channels is not None:
            self.out_split_size = out_channels
            out_channels = out_channels + additional_out_channels


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        #PAD input
        padded_input = self.input_pad(x)
        conv1 = self.conv1(padded_input)
        out = self.conv2(conv1)

        if self.out_split_size is not None:
            return out[:, :self.out_split_size, ...], out[:, self.out_split_size:, ...]
        return out



def compute_backbone_feature_shape(backbone):
    '''
    Computes the feature space output dimensions based on input image shape
    :return: shape of the feature space vector
    '''
    t = torch.randn(cfg.INPUT_IMAGE_SHAPE)
    # out = self.forward(t)
    out = backbone(t)
    out_shape = out.shape
    return out_shape

def build_MLP(n_in, output=None,
              multiple_output=None,
              hidden_layers = cfg.DEFAULT_MLP_TOPOLOGY,
              activation = None,
              internal_activation = ReLU):
    '''
    builds a MLP that supports multiple outputs when 'multiple_outputs' are specified
    :param n_in: input size
    :param out: output dimension, if a tuple/list, then it will
    :param hidden_layers: hidden layers
    :param activation: output activation function
    :param internal_activation: internal activation function
    :return:
    '''

    n_prev = n_in
    net = OrderedDict()

    # build hidden layers
    for i, h in enumerate(hidden_layers):
        net['dense%d' % i] = Linear(n_prev, h)
        net['relu%d' % i] = internal_activation()
        n_prev = h

    # build output layer


    # Output is singular
    if output is not None:
        net['out'] = Linear(n_prev, output)
        if activation is not None:
            net['act'] = activation()
        return Sequential(net)
    # Output is multiple
    elif multiple_output is not None:
        out_net = OrderedDict()
        for i, out in enumerate(multiple_output):
            out_net['out_%d' % i] = Linear(n_prev, out)

        return SequentialMultipleOutput(net, out_net)
    else:
        raise AssertionError('Unknown output type')

def latent_to_mean_std(latent_var):
    '''
    Converts a VAE latent vector to mean and std. log_std is converted to std.
    :param latent_var: VAE latent vector
    :return:
    '''
    mean, log_std = torch.chunk(latent_var, 2, dim=1)
    # std = log_std.mul(0.5).exp_()
    std = torch.sigmoid(log_std.clamp(-10, 10)) * 2
    return mean, std

def clamped_sigmoid(logit, use_analytical=False):
    '''
    Sigmoid function,
    :param logit:
    :param use_analytical: use analytical sigmoid function to prevent backprop issues in pytorch
    :return:
    '''
    # logit = torch.clamp(logit, -10, 10)
    if use_analytical:
        logit = torch.clamp(logit, -10, 10)
        return 1 / ((-logit).exp() + 1)

    return torch.sigmoid(torch.clamp(logit, -10, 10))

def exponential_decay(start, end, decay_rate, decay_step:float, staircase=False, log_space=False, ):
    '''
    A decay helper function for computing decay of
    :param global_step:
    :param start:
    :param end:
    :param decay_rate:
    :param decay_step:
    :param staircase:
    :param log_space:
    :return:
    '''

    global_step = torch.tensor(RunManager.global_step, dtype=torch.float32).to(RunManager.device)
    if staircase:
        t = global_step // decay_step
    else:
        t = global_step / decay_step
    value = (start - end) * (decay_rate ** t) + end

    if log_space:
        value = (value + 1e-6).log()

    return value


def stn(image, z_where, output_dims, inverse=False):
    """
    Slightly modified based on https://github.com/kamenbliznashki/generative_models/blob/master/air.py

    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """

    xt, yt, xs, ys = torch.chunk(z_where, 4, dim=-1)
    yt = yt.squeeze()
    xt = xt.squeeze()
    ys = ys.squeeze()
    xs = xs.squeeze()

    batch_size = image.shape[0]
    color_chans = cfg.INPUT_IMAGE_SHAPE[0]
    out_dims = [batch_size, color_chans] + output_dims # [Batch, RGB, obj_h, obj_w]

    # Important: in order for scaling to work, we need to convert from top left corner of bbox to center of bbox
    yt = (yt ) * 2 - 1
    xt = (xt ) * 2 - 1

    theta = torch.zeros(2, 3).repeat(batch_size, 1, 1).to(RunManager.device)

    # set scaling
    theta[:, 0, 0] = xs
    theta[:, 1, 1] = ys
    # set translation
    theta[:, 0, -1] = xt
    theta[:, 1, -1] = yt

    # inverse == upsampling
    if inverse:
        # convert theta to a square matrix to find inverse
        t = torch.tensor([0., 0., 1.]).repeat(batch_size, 1, 1).to(RunManager.device)
        t = torch.cat([theta, t], dim=-2)
        t = t.inverse()
        theta = t[:, :2, :]
        out_dims = [batch_size, color_chans + 1] + output_dims  # [Batch, RGBA, obj_h, obj_w]

    # 2. construct sampling grid
    grid = F.affine_grid(theta, out_dims)

    # 3. sample image from grid
    padding_mode = 'border' if not inverse else 'zeros'
    input_glimpses = F.grid_sample(image, grid, padding_mode=padding_mode)
    # debug_tools.plot_stn_input_and_out(input_glimpses)


    return input_glimpses


class SequentialMultipleOutput(Module):
    def __init__(self, input, outputs):
        super().__init__()
        self.body = Sequential(input)
        self.output_layers = ModuleList(list(outputs.values()))

    def forward(self, x):
        body_out = self.body(x)
        return (output_layer(body_out) for output_layer in self.output_layers)

def to_C_H_W(t:torch.Tensor):
    # From [B, H, W, C] to [B, C, H, W]
    assert t.shape[1] == t.shape[2] and t.shape[3] != t.shape[2], 'are you sure this tensor is in [B, H, W, C] format?'
    return t.permute(0,3,1,2)

def to_H_W_C(t:torch.Tensor):
    # From [B, C, H, W] to [B, H, W, C]
    assert t.shape[2] == t.shape[3] and t.shape[1] != t.shape[2], 'are you sure this tensor is in [B, C, H, W] format?'
    return t.permute(0,2,3,1)

def safe_log(t):
    return torch.log(t + 1e-9)