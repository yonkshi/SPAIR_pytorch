import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, Linear
from torch.nn import functional as F
from torch.distributions import Normal, Uniform
from torch.distributions.kl import kl_divergence
from spair import config as cfg
from spair.modules import *

class SPAIR(nn.Module):
    def __init__(self, image_shape ):
        super().__init__()
        self.image_shape = image_shape
        # self.pixels_per_cell = pixels_per_cell
        self.B = 1 # TODO Change num bounding boxes

        # Keeps track of all distributions
        self.dist = {}
        # object presence probability
        self.obj_pres_prob = {}
        self.obj_pres = {}

        # context box dimension based on N_lookbacks
        # totalsize is n_lookback_cells * [localization, attribute, depth, presence] ~= 224
        self.context_dim = (cfg.N_LOOKBACK * 2 + 1 ) ** 2 // 2 * (4 + cfg.N_ATTRIBUTES + 1 + 1)

        self._build_networks()
        self._build_edge_element()
        self._build_indep_prior()

        self.pixels_per_cell = tuple(int(i) for i in self.backbone.grid_cell_size) #[pixel_x, pixel_y]
        print('model initialized')

    def forward(self, x, global_step = 0):
        # feature space
        _, H, W = self.feature_space_dim
        feat = self.backbone(x)
        self.global_step = global_step
        self.batch_size = x.shape[0]
        context_mat = {}

        z_where = torch.empty(self.batch_size, 4, H, W) # 4 = xt, yt, xs, ys
        z_attr = torch.empty(self.batch_size, cfg.N_ATTRIBUTES, H, W,)
        z_depth = torch.empty(self.batch_size, 1, H, W)
        z_pres = torch.empty(self.batch_size, 1, H, W)
        z_pres_prob = torch.empty(self.batch_size, 1, H, W)

        edge_element = self.virtual_edge_element[None,:].repeat(self.batch_size, 1)
        self.training_wheel = exponential_decay(self.global_step, **cfg.LATENT_VAR_TRAINING_WHEEL_PARAM)

        # Iterate through each grid cell and bounding boxes for that cell
        for h, w in itertools.product(range(H), range(W)):

            # feature vec for each cell in the grid
            cell_feat = feat[:, :, h, w]

            context = self._get_sequential_context(context_mat, h,w, edge_element)

            # --- box ---
            layer_inp = torch.cat((cell_feat, context), dim=-1)
            rep_input, passthru_features = self.box_network(layer_inp)
            box, normalized_box = self._build_box(rep_input, h, w)
            z_where[:, :, h, w,] = normalized_box

            # --- attr ---
            input_glimpses, attr_latent_var = self._encode_attr(x, normalized_box)
            attr_mean, attr_std = latent_to_mean_std(attr_latent_var)
            attr = self._sample_z(attr_mean, attr_std, 'attr', (h, w))
            z_attr[:, :, h, w, ] = attr

            # --- depth ---
            layer_inp = torch.cat([cell_feat, context, passthru_features, box, attr], dim=1)

            depth_latent, passthru_features = self.z_network(layer_inp)

            depth_mean, depth_std = latent_to_mean_std(depth_latent)
            depth_mean, depth_std = self._freeze_learning(depth_mean, depth_std)

            depth_logits = self._sample_z(depth_mean, depth_std, 'depth_logit', (h,w))
            depth = 4 * clamped_sigmoid(depth_logits)
            z_depth[:,:, h, w] = depth

            # --- presence ---
            layer_inp = torch.cat([cell_feat, context, passthru_features, box, attr, depth], dim=1)
            pres_logit = self.obj_network(layer_inp)
            obj_pres, obj_pres_prob = self._build_obj_pres(pres_logit)

            z_pres[:,:, h,w] = obj_pres
            z_pres_prob[:, :, h, w] = obj_pres_prob

            context_mat[(h,w)] = torch.cat((box, attr, depth, obj_pres), dim=-1)

        # Merge dist param, we have to use loop or autograd might not work
        for dist_name, dist_params in self.dist.items():
            means = self.dist[dist_name]['mean']
            sigmas = self.dist[dist_name]['sigma']
            self.dist[dist_name] = Normal(loc=means, scale=sigmas)

        self._compute_KL(z_pres, z_pres_prob)
        self._render(z_attr, z_where, z_depth, z_pres)

    def _compute_KL(self, z_pres, z_pres_prob):
        KL = {}
        # For all latent distributions
        for dist_name, dist_params in self.dist.items():
            dist = self.dist[dist_name]
            prior = self.kl_priors[dist_name]
            KL[dist_name] = kl_divergence(dist, prior)

        # --- Object Presence KL ---
        # Special prior computation, refer to Appendix B for detail
        _, H, W = self.feature_space_dim
        HW = H * W
        batch_size = self.batch_size # TODO change batch size
        count_support = torch.arange(HW + 1, dtype=torch.float32) # [50] ~ [0, 1, ... 50]
        # FIXME starts at 1 output and gets small gradually
        count_prior_log_odds = exponential_decay(self.global_step, **cfg.OBJ_PRES_COUNT_LOG_PRIOR)
        count_prior_prob = torch.sigmoid(count_prior_log_odds)
        count_distribution = (1 - count_prior_prob) * (count_prior_prob ** count_support) # [50]

        normalizer = count_distribution.sum()
        count_distribution = count_distribution / normalizer
        count_distribution = count_distribution.repeat(batch_size, 1)  # (Batch, 50)

        # number of symbols discovered so far
        count_so_far = torch.zeros(batch_size, 1) # (Batch, 1)

        i = 0

        obj_kl = []

        for h, w in itertools.product(range(H), range(W)):
            p_z_given_Cz = torch.clamp(count_support - count_so_far, max=0) / (HW - i)

            # Reshape for batch matmul
            # Adds a new dim to to each vector for matrix multiplication [Batch, 50, ?]
            _count_distribution = count_distribution[:, None, :]
            _p_z_given_Cz = p_z_given_Cz[:, :, None]
            # Computing the prior, flatten tensors [Batch, 1]
            p_z = (_count_distribution * _p_z_given_Cz)[:, :, 0]

            prob = z_pres_prob[:, :, h, w]

            # Bernoulli KL
            # note to self: May need to use safe log to prevent NaN
            _obj_kl = (
                prob * (torch.log(prob) - torch.log(p_z))
                + (1-prob) * (torch.log(1-prob) - torch.log(1-p_z))
            )

            obj_kl.append(_obj_kl)

            # Check if object presents (0.5 threshold)
            # original: tf.to_float(tensors["obj"][:, h, w, b, :] > 0.5), but obj should already be rounded
            sample = torch.round(z_pres[:, :, h, w])
            # Bernoulli prob
            mult = sample * p_z_given_Cz + (1-sample) * (1-p_z_given_Cz)

            # update count distribution
            # FIXME why multiplying mult
            count_distribution = mult * count_distribution
            normalizer = count_distribution.sum(dim=1, keepdim=True)
            # why clip normalizer?
            normalizer = torch.clamp(normalizer, max=1e-6)
            count_distribution = count_distribution / normalizer

            count_so_far += sample

            i += 1


        KL['pres_dist'] = obj_kl


        return KL

    def _build_indep_prior(self):
        '''
        builds independ priors for the VAE for KL computation (VAE KL Regularizer)
        '''
        self.kl_priors = {}
        for z_name, (mean, std) in cfg.PRIORS.items():
            dist = Normal(mean, std)
            self.kl_priors[z_name] = dist

    def _build_edge_element(self):
        '''
            This method builds the learnable virtual cell for context building,
         context requires 4 surrounding cells, if they fall outside grid, then this cell is used
         '''

        # sizes for localization [x, y, h, w], obj attributes, obj depth, obj presence
        sizes = [4, cfg.N_ATTRIBUTES, 1, 1]
        t = torch.empty(sum(sizes), requires_grad=True)
        loc, attr, depth, pres = torch.split(t, sizes)

        loc = torch.nn.Sigmoid()(loc)
        pres = torch.nn.Sigmoid()(pres)

        self.virtual_edge_element = torch.cat((loc, attr, depth, pres))

    def _build_networks(self):

        # backbone network
        self.backbone = Backbone(self.image_shape, cfg.N_BACKBONE_FEATURES)
        self.feature_space_dim = self.backbone.compute_output_shape()

        n_passthrough_features = cfg.N_PASSTHROUGH_FEATURES
        n_localization_latent = 8  # mean and var for (y, x, h, w)
        n_backbone_features = self.feature_space_dim[0]

        # bounding box
        inputsize = n_backbone_features + self.context_dim
        self.box_network = build_MLP(inputsize, multiple_output=(n_localization_latent, n_passthrough_features))

        # object attribute
        n_attr_out = 2 * cfg.N_ATTRIBUTES
        obj_dim = cfg.OBJECT_SHAPE[0]
        n_inp_shape = obj_dim * obj_dim * 3 # flattening the 14 x 14 x 3 image
        self.object_encoder = build_MLP(n_inp_shape, n_attr_out, hidden_layers=[256, 128])

        # object depth
        z_inp_shape = 4 + cfg.N_ATTRIBUTES + n_passthrough_features + self.context_dim + cfg.N_BACKBONE_FEATURES
        self.z_network = build_MLP(z_inp_shape, multiple_output=(2, n_passthrough_features)) # For training pass through features

        # object presence
        obj_inp_shape = z_inp_shape + 1 # incorporated z_network out dim (1 dim)
        self.obj_network = build_MLP(obj_inp_shape, 1)

        # object decoder
        decoded_dim = obj_dim * obj_dim * 4 #
        self.object_decoder = build_MLP(cfg.N_ATTRIBUTES, decoded_dim, hidden_layers=[128, 256])

    def _get_sequential_context(self, context_mat:dict, h, w, edge_element):

        range = cfg.N_LOOKBACK

        # build a range of nearby visited rows and cols. Rows include current row
        cols = np.arange(-range, range+1)
        rows = np.arange(-range, 1)
        # generate all neighbouring cells above current cell
        mesh = np.array(np.meshgrid(rows, cols)).T
        # flatten the coordinates to N x 2
        flattened = np.reshape(mesh, (-1, 2))
        # remove the last few elements that were inlucded accidentally
        coords = flattened[:-(range+1), :]

        # relative coords to absolute coords
        neighbours = coords + np.array([h, w], dtype=np.float32)
        context = []

        # TODO Might be able to get rid of this loop, replace with faster ops
        for coord in neighbours:
            coord = tuple(coord)
            if context_mat and coord in context_mat.keys():
                context.append(context_mat[coord])
            else:
                context.append(edge_element)

        context = torch.cat(context, dim=-1)

        return context

    def _build_box(self, latent_var, h, w):
        ''' Builds the bounding box from latent space z'''
        # TODO Double check if I can replace exp with sigmoid
        mean, std = latent_to_mean_std(latent_var)

        mean, std = self._freeze_learning(mean, std)

        #
        cy_mean, cx_mean, height_mean, width_mean = torch.chunk(mean, 4, dim=-1)
        cy_std, cx_std, height_std, width_std = torch.chunk(std, 4, dim=-1)

        cy_logits = self._sample_z(cy_mean, cy_std, 'cy_logit', (h,w))
        cx_logits = self._sample_z(cx_mean, cx_std, 'cx_logit', (h,w))
        height_logits = self._sample_z(height_mean, height_std, 'height_logit', (h,w))
        width_logits = self._sample_z(width_mean, width_std, 'width_logit', (h, w))

        # --- cell y/x transform ---
        cell_y = clamped_sigmoid(cy_logits) # single digit
        cell_x = clamped_sigmoid(cx_logits)

        # yx ~ [-0.5 , 1.5]
        max_yx = cfg.MAX_YX
        min_yx = cfg.MIN_YX
        assert max_yx > min_yx
        cell_y = float(max_yx - min_yx) * cell_y + min_yx
        cell_x = float(max_yx - min_yx) * cell_x + min_yx

        # --- height/width transform ---
        height = clamped_sigmoid(height_logits)
        width = clamped_sigmoid(width_logits)
        max_hw = cfg.MAX_HW
        min_hw = cfg.MIN_YX
        assert max_hw > min_hw
        # hw ~ [0.0 , 1.0]
        # current bounding box height & width ratio to anchor box
        height = float(max_hw - min_hw) * height + min_hw
        width = float(max_hw - min_hw) * width + min_hw

        box = torch.cat([cell_y, cell_x, height, width], dim=-1)

        # --- Compute image-normalized box parameters ---

        # box height and width normalized to image height and width
        anchor_box_dim = cfg.ANCHORBOX_SHAPE[0]
        _, image_height, image_width = cfg.INPUT_IMAGE_SHAPE
        # bounding box height & width relative to the whole image
        ys = height * anchor_box_dim / image_height
        xs = width * anchor_box_dim / image_width

        # box centre normalized to image height and width
        yt = (self.pixels_per_cell[0] / image_height) * (cell_y + h)
        xt = (self.pixels_per_cell[1] / image_width) * (cell_x + w)

        yt -= ys / 2.
        xt -= xs / 2.

        normalized_box = torch.cat([yt, xt, ys, xs], dim=-1)

        return box, normalized_box

    def _encode_attr(self, x, normalized_box):
        ''' Uses spatial transformation to crop image '''
        # --- Get object attributes using object encoder ---

        input_glimpses = stn(x, normalized_box, cfg.OBJECT_SHAPE)
        flat_input_glimpses = input_glimpses.flatten(start_dim=1) # flatten
        attr = self.object_encoder(flat_input_glimpses)
        # attr = attr.view(-1, 2 * cfg.N_ATTRIBUTES)
        return input_glimpses, attr

    def _build_obj_pres(self, obj_logits):
        ''' Builds the network to detect object presence'''
        obj_logits = self._freeze_learning(obj_logits)
        # obj_logits = obj_logits / self.obj_temp

        obj_log_odds = torch.clamp(obj_logits, -10., 10.)

        # Adding relative noise to object presence, possibly to prevent trivial mapping?
        eps = 10e-10
        u = Uniform(0, 1)
        u = u.rsample(obj_log_odds.size())
        noise = torch.log(u + eps) - torch.log(1.0 - u + eps)
        obj_pre_sigmoid = (obj_log_odds + noise) / 1.0

        obj_prob = torch.sigmoid(obj_pre_sigmoid)  # Object Classification
        # TODO If training then pass through, else round it up
        obj = obj_prob # torch.round(obj)

        return obj, obj_prob

    def _freeze_learning(self, *args):
        '''
        Some latent variables needs to be frozen during the early stages of training
        :param mean:
        :param var:
        :return:
        '''

        freeze_factor = self.training_wheel

        ret = []
        for arg in args:
            arg = freeze_factor * arg.detach() + (1 - freeze_factor) * arg
            ret.append(arg)

        if len(ret) == 1: return ret[0] # single element doesn't need unfolding
        return ret

    def _sample_z(self, mean, var, name, cell_coord):
        '''
        Performs the sampling step in VAE and stores the distribution for KL computation
        :param mean:
        :param var:
        :param name: name of the distribution
        :return: sampled value
        '''
        dist = Normal(loc=mean, scale=var)

        if name not in self.dist.keys():
            _, H, W = self.feature_space_dim
            self.dist[name] = {}
            self.dist[name]['mean'] = torch.empty(self.batch_size, H, W, mean.shape[-1])
            self.dist[name]['sigma'] = torch.empty(self.batch_size, H, W, mean.shape[-1])
        x, y = cell_coord
        self.dist[name]['mean'][:, x, y, :] = mean
        self.dist[name]['sigma'][:, x, y, :] = var

        return dist.rsample()

    def _render(self, z_attr, z_where, z_depth, z_pres):
        _, H, W = self.feature_space_dim
        px = cfg.OBJECT_SHAPE[0]

        # ---- Now entering B x H x W x C realm, because we needed to merge B*H*W ----


        # Permute then flatten latent vars from [Batch, ?, H, W, ?] to [Batch * H * W, ?] so we can feed into STN
        z_where = to_H_W_C(z_where).contiguous().view(-1, 4)


        # flattening z depth and z presence for element-wise broadcasted multiplication later
        z_depth = z_depth.view(-1, 1, 1)
        z_pres = z_pres.view(-1, 1, 1)

        object_decoder_in = to_H_W_C(z_attr).contiguous().view(-1, cfg.N_ATTRIBUTES)

        # MLP to generate image
        object_logits = self.object_decoder(object_decoder_in)
        object_logits = object_logits.view(-1, px, px, 4) # [Batch * n_cells, pixels_w, pixels_h, channels]

        # object_logits scale + bias mask
        object_logits[:, :, :, :-1] *= cfg.OBJ_LOGIT_SCALE  #[B, 14, 14, 4] * [4]
        object_logits[:, :, :, -1] *= object_logits[:, :, :, -1] * cfg.ALPHA_LOGIT_SCALE + cfg.ALPHA_LOGIT_BIAS

        objects = clamped_sigmoid(object_logits)
        objects = objects.view(-1, px, px, 4,)

        # incorporate presence in alpha channel
        objects[:,:,:,-1] *= z_pres.expand_as(objects[:,:,:,-1]) # Todo double check if broadcasting works

        # importance manipulates how gradients scales, but does not nessasarily
        importance = objects[:, :, :, -1] * z_depth.expand_as(objects[:,:,:,-1])
        importance = torch.clamp(importance, max=0.01) # not sure why clamp it

        # ---- exiting B x H x W x C realm .... ----

        objects = to_C_H_W(objects)

        img_c, img_h, img_w, = (self.image_shape)

        transformed_imgs = stn(objects, z_where, [img_h, img_w])

        # incorporate alpha
        # TODO Blend alpha
        transformed_objects = transformed_imgs.contiguous().view(self.batch_size ,H, W, 4, img_h, img_w)
        # scale gradient
        importance = importance.view(self.batch_size,H, W)
        importance = importance.expand_as(transformed_objects)
        gradient_scaled_objects = transformed_objects * importance + (1 - importance) * transformed_objects.detach()

        gradient_scaled_objects.sum(dim=(1,2)) # sum up H and W to [Batch, image, 3


        pass


