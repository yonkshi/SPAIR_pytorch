import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F
import numpy as np

class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2


    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        zk = zk.squeeze(1)
        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = w * u
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = w ** 2
        u_hat = u + ((m_uw - uw) * w / w_norm_sq)

        # compute flow with u_hat
        wzb = w*zk + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + psi * u_hat))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian

class SingleZPlanarNF2d(nn.Module):
    """
    One dimensional Normalizing flow (Planar flow)
    """

    def __init__(self, num_flows, q_z_nn_dim):
        super().__init__()

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = Planar
        self.num_flows = num_flows
        self.h_dim = q_z_nn_dim
        self.z_size = 1

        # Amortized flow parameters
        self.amor_u = nn.Conv2d(self.h_dim, self.num_flows * self.z_size, kernel_size=1, stride=1)
        self.amor_w = nn.Conv2d(self.h_dim, self.num_flows * self.z_size, kernel_size=1, stride=1)
        self.amor_b = nn.Conv2d(self.h_dim, self.num_flows, kernel_size=1, stride=1)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def forward(self, z,  h):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        u = self.amor_u(h)
        w = self.amor_w(h)
        b = self.amor_b(h)

        # first z_k
        z_k = z

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z_k, u[:, k, ...], w[:, k, ...], b[:, k, ...])
            self.log_det_j += log_det_jacobian

        z_k = z_k.unsqueeze(1)
        self.log_det_j = self.log_det_j.unsqueeze(1)
        return z_k, self.log_det_j





def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()
class MaskedConv2d(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1)

        self.register_buffer('mask', mask)

    def forward(self, inputs):
        mask_broadcast = self.mask[..., None, None]
        output = F.conv2d(inputs, self.conv.weight * mask_broadcast,
                          self.conv.bias)

        return output

class MADEConv(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 act='relu',
                 pre_exp_tanh=False):
        super().__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = MaskedConv2d(num_inputs, num_hidden, input_mask)

        self.trunk = nn.Sequential(act_func(),
                                   MaskedConv2d(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   MaskedConv2d(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs)
            trunk = self.trunk(h)
            m, a = trunk.chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(1, keepdim=True).mean(dim=[-1, -2]) # Mean of the h and w

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(1, keepdim=True).mean(dim=[-1, -2])

class BatchNormFlow2d(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super().__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(dim=[0,2,3]) # mean along batch, h an w
                self.batch_var = (
                    inputs - self.batch_mean[-1, None, None]).pow(2).mean(dim=[0,2,3]) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean[-1, None, None]) / var[-1, None, None].sqrt()
            y = torch.exp(self.log_gamma)[-1, None, None] * x_hat + self.beta[-1, None, None]
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)

class Reverse2d(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super().__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm, ...], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)

class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None, use_conv = False):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None, use_conv = False):
        # FIXME Calling forward
        u, log_jacob = self(inputs, cond_inputs, use_conv=use_conv)
        log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum(
            1, keepdim=True).mean(dim=[-1,-2])
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples