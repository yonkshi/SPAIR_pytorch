import itertools

from torch import nn
from torch.distributions import Normal, Uniform
from torch.distributions.kl import kl_divergence
from spair.modules import *
from spair.manager import RunManager
from spair.modules import Backbone, build_MLP


class SpairBase(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape
        self.writer = RunManager.writer
        # self.pixels_per_cell = pixels_per_cell
        self.device = RunManager.device

        # context box dimension based on N_lookbacks
        # totalsize is n_lookback_cells * [localization, attribute, depth, presence] ~= 224
        self.context_dim = (cfg.N_LOOKBACK * 2 + 1 ) ** 2 // 2 * (4 + cfg.N_ATTRIBUTES + 1 + 1)

        self._build_networks()
        self._build_edge_element()
        self._build_indep_prior()

        self.pixels_per_cell = tuple(int(i) for i in self.backbone.grid_cell_size) #[pixel_x, pixel_y]

        self.run_args = RunManager.run_args
        print('model initialized')

    def forward(self, x):
        debug_tools.benchmark_init()
        # feature space
        _, H, W = self.feature_space_dim
        backbone_feat = self.backbone(x)
        self.global_step = RunManager.global_step
        self.batch_size = x.shape[0]

        # Keeps track of all distributions
        self.dist_param = {}
        self.dist = {}
        # object presence probability
        self.obj_pres_prob = {}
        self.obj_pres = {}


        self._compute_latent_vars(x, backbone_feat)

        # self.attn(test_context)

        kl_loss = self._compute_KL(self.z_pres, self.z_pres_prob)
        recon_x = self._render(self.z_attr, self.z_where, self.z_depth, self.z_pres, x)
        loss = self._build_loss(x, recon_x, kl_loss)

        return loss, recon_x, self.z_where, self.z_pres

    def _compute_KL(self, z_pres, z_pres_prob):
        KL = {}
        # For all latent distributions
        for dist_name, dist_params in self.dist.items():
            dist = self.dist[dist_name]
            prior = self.kl_priors[dist_name]
            kl_div = kl_divergence(dist, prior)
            masked = z_pres * kl_div
            KL[dist_name] = masked

        # TODO Temporary z_presence test
        # Disable z_presence prior
        if self.run_args.no_z_prior:
            return KL

        # --- Object Presence KL ---
        # Special prior computation, refer to Appendix B for detail
        _, H, W = self.feature_space_dim
        HW = H * W
        batch_size = self.batch_size
        count_support = torch.arange(HW + 1, dtype=torch.float32).to(self.device)# [50] ~ [0, 1, ... 50]
        # FIXME starts at 1 output and gets small gradually
        count_prior_log_odds = exponential_decay(**cfg.OBJ_PRES_COUNT_LOG_PRIOR)
        # count_prior_prob = torch.sigmoid(count_prior_log_odds)
        count_prior_prob = 1 / ((-count_prior_log_odds).exp() + 1)
        # p(z_pres|C=nz(z_pres)) geometric dist, see appendix A
        count_distribution = (1 - count_prior_prob) * (count_prior_prob ** count_support)

        normalizer = count_distribution.sum()
        count_distribution = count_distribution / normalizer
        count_distribution = count_distribution.repeat(batch_size, 1)  # (Batch, 50)

        # number of symbols discovered so far
        count_so_far = torch.zeros(batch_size, 1).to(self.device) # (Batch, 1)

        i = 0

        obj_kl = torch.ones(batch_size, 1, H, W).to(self.device)

        # print('\n\np_z_given_Cz%d'%i, p_z_given_Cz, torch.isnan(p_z_given_Cz).sum())
        for h, w in itertools.product(range(H), range(W)):

            p_z_given_Cz = torch.clamp(count_support - count_so_far, min=0., max=(HW - i)) / (HW - i)

            # Reshape for batch matmul
            # Adds a new dim to to each vector for dot product [Batch, 50, ?]
            _count_distribution = count_distribution[:,None,:]
            _p_z_given_Cz = p_z_given_Cz[:,:,None]


            prob = z_pres_prob[:, :, h, w]

            # TODO This is for testing uniform dist
            if self.run_args.uniform_z_prior:
                p_z = torch.ones_like(prob) / HW
            else:
                # Computing the prior, flatten tensors [Batch, 1]
                # equivalent of doing batch dot product on two vectors
                p_z = torch.bmm(_count_distribution, _p_z_given_Cz).squeeze(-1)


            # Bernoulli KL
            # note to self: May need to use safe log to prevent NaN
            _obj_kl = (
                prob * (safe_log(prob) - safe_log(p_z))
                + (1-prob) * (safe_log(1-prob) - safe_log(1-p_z))
            )

            obj_kl[:, :, h, w] = _obj_kl

            # Check if object presents (0.5 threshold)
            # original: tf.to_float(tensors["obj"][:, h, w, b, :] > 0.5), but obj should already be rounded
            sample = torch.round(z_pres[:, :, h, w])
            # Bernoulli prob
            mult = sample * p_z_given_Cz + (1-sample) * (1-p_z_given_Cz)

            # update count distribution
            count_distribution1 = mult * count_distribution
            normalizer = count_distribution1.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # why clip normalizer?
            # normalizer = torch.clamp(normalizer, min=1e-6)
            count_distribution = count_distribution1 / normalizer

            # Test underflow issues

            debug_tools.nan_hunter('KL Divergence',
                                   grid_location = (h,w),
                                   _obj_kl = _obj_kl,
                                   p_z = p_z,
                                   prob = prob,
                                   count_distribution = count_distribution,
                                   p_z_given_cz = p_z_given_Cz,
                                   )

            count_so_far += sample

            i += 1


        KL['pres_dist'] = obj_kl

        return KL

    def _compute_latent_vars(self, x, backbone_features):
        raise NotImplementedError()

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
        t = torch.randn(sum(sizes), requires_grad=True).to(self.device)
        loc, attr, depth, pres = torch.split(t, sizes)

        loc = torch.nn.Sigmoid()(loc)
        pres = torch.nn.Sigmoid()(pres)
        depth = torch.nn.Sigmoid()(depth)

        elem = torch.cat((loc, attr, depth, pres))
        self.register_parameter('virtual_edge_element', nn.Parameter(elem))

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
        u = u.rsample(obj_log_odds.size()).to(self.device)
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

    def _render(self, z_attr, z_where, z_depth, z_pres, x):
        '''
        decoder + renderer function. combines the latent vars & bbox, feed into the decoder for each object and then
        :param z_attr:
        :param z_where:
        :param z_depth:
        :param z_pres:
        :return:
        '''
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


        input_chan_w_alpha = cfg.INPUT_IMAGE_SHAPE[0] + 1
        object_logits = object_logits.view(-1, px, px, input_chan_w_alpha) # [Batch * n_cells, pixels_w, pixels_h, channels]


        # object_logits scale + bias mask
        object_logits[:, :, :, :-1] *= cfg.OBJ_LOGIT_SCALE  #[B, 14, 14, 4] * [4]
        object_logits[:, :, :, -1] *= cfg.ALPHA_LOGIT_SCALE
        object_logits[:, :, :, -1] += cfg.ALPHA_LOGIT_BIAS
        # TODO DELETE ME
        # z_attr.register_hook(lambda grad: debug_tools.z_attr_grad_hook(grad, self.writer, self.global_step))
        # object_logits.register_hook(lambda grad: debug_tools.decoder_output_grad_hook(grad, self.writer, self.global_step))
        # TODO END DELETE ME
        objects = clamped_sigmoid(object_logits, use_analytical=True)
        objects = objects.view(-1, px, px, input_chan_w_alpha)

        # incorporate presence in alpha channel
        objects[:,:,:,-1] *= z_pres.expand_as(objects[:,:,:,-1])

        # importance manipulates how gradients scales, but does not nessasarily
        importance = objects[:, :, :, -1] * z_depth.expand_as(objects[:,:,:,-1])
        importance = torch.clamp(importance, min=0.01)

        # Merge importance to objects:
        importance = importance[..., None] # add a trailing dim for concatnation
        objects = torch.cat([objects, importance], dim=-1 ) # attach importance to RGBA, 5 channels to total

        # TODO debug output image, remove me later
        debug_tools.plot_prerender_components(objects, z_pres, z_depth, z_where, x)

        # ---- exiting B x H x W x C realm .... ----

        objects_ = to_C_H_W(objects)

        img_c, img_h, img_w, = (self.image_shape)
        n_obj = H*W # max number of objects in a grid
        transformed_imgs = stn(objects_, z_where, [img_h, img_w], inverse=True)
        transformed_objects = transformed_imgs.contiguous().view(-1, n_obj, img_c + 2 , img_h, img_w)
        # incorporate alpha
        # FIXME The original implement doesn't seem to be calculating alpha correctly.
        #  If multiple objects overlap one pixel, alpha is only computed against background
        # TODO assume background is black. Will learn to construct different background in the future

        # TODO we can potentially compute alpha and importance prior to stn, it will be much faster

        input_chan = cfg.INPUT_IMAGE_SHAPE[0]
        color_channels  = transformed_objects[:, :, :input_chan, :, :]
        alpha = transformed_objects[:, :, input_chan:input_chan+1, :, :] # keep the empty dim
        importance = transformed_objects[:,:, input_chan+1:input_chan+2, :, :] + 1e-9

        img = alpha.expand_as(color_channels) * color_channels

        # normalize importance
        importance = importance / importance.sum(dim=1, keepdim=True)
        importance = importance.expand_as(img)
        # scale gradient
        weighted_grads_image = img * importance # + (1 - importance) * img.detach()

        output_image = weighted_grads_image.sum(dim=1) # sum up along n_obj per image

        # Fix numerical issue
        output_image = torch.clamp(output_image, min=0, max=1)

        return output_image

    def _build_loss(self, x, recon_x, kl):
        # print('============ Losses =============')
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        self.writer.add_scalar('losses/reconst', recon_loss, self.global_step)
        # print('Reconstruction loss:', '{:.4f}'.format(recon_loss.item()))
        # KL loss with Beta factor
        kl_loss = 0
        for name, z_kl in kl.items():
            kl_mean = torch.mean(torch.sum(z_kl, dim=[1,2,3])) # batch mean
            kl_loss += kl_mean
            # print('KL_%s_loss:' % name, '{:.4f}'.format(kl_mean.item()))
            self.writer.add_scalar('losses/KL{}'.format(name), kl_mean, self.global_step )

        loss = recon_loss + cfg.VAE_BETA * kl_loss
        # print('\n ===> total loss:', '{:.4f}'.format(loss.item()))
        self.writer.add_scalar('losses/total', loss, self.global_step)


        return loss

    def _debug_logging(self, z_where, z_attr, z_pres, z_depth ):

        if self.training_wheel == 1.0: # only start training when training wheel enables
            return
        z_where = z_where.cpu().detach()
        z_pres = z_pres.cpu().detach()
        z_depth = z_depth.cpu().detach()

        _, H, W = self.feature_space_dim
        h, w = np.random.randint(W, size=2)
        np.set_printoptions(threshold=np.inf, precision=6)

        print('=========== debugging information begin ===============')
        box = z_where[0, :, h , w].numpy()
        box_x, box_y, box_w, box_h = box
        print('box:')
        print(' x:\t{:.6f}'.format(box_x))
        print(' y:\t{:.6f}'.format(box_y))
        print(' w:\t{:.6f}'.format(box_w))
        print(' h:\t{:.6f}'.format(box_h))

        self.writer.add_histogram('box/x', z_where[0, 0, ...], self.global_step)
        self.writer.add_histogram('box/y', z_where[0, 1, ...], self.global_step)
        self.writer.add_histogram('box/w', z_where[0, 2, ...], self.global_step)
        self.writer.add_histogram('box/h', z_where[0, 3, ...], self.global_step)
        print('')
        # z_presence
        z_pres_np = z_pres[0, ...].numpy()
        print('z_pres:', z_pres_np)
        self.writer.add_scalar('z_presence/max', z_pres_np.max(), self.global_step)
        self.writer.add_scalar('z_presence/mean', z_pres_np.mean(), self.global_step)
        self.writer.add_scalar('z_presence/min', z_pres_np.min(), self.global_step)
        print('')
        # z_depth
        z_depth_np = z_depth[0, ...].numpy()
        print('z_depth:', z_depth_np)
        self.writer.add_scalar('z_depth/max', z_depth_np.max(), self.global_step)
        self.writer.add_scalar('z_depth/mean', z_depth_np.mean(), self.global_step)
        self.writer.add_scalar('z_depth/min', z_depth_np.min(), self.global_step)
        print('========= end of debugging info  ======================')

class Spair(SpairBase):

    def _compute_latent_vars(self, x, backbone_features):



        _, H, W = self.feature_space_dim
        context_mat = {}
        edge_element = self.virtual_edge_element[None, :].repeat(self.batch_size, 1)
        self.training_wheel = exponential_decay(**cfg.LATENT_VAR_TRAINING_WHEEL_PARAM)
        self.writer.add_scalar('training_wheel', self.training_wheel, self.global_step)

        self.z_where = torch.empty(self.batch_size, 4, H, W).to(self.device) # 4 = xt, yt, xs, ys
        self.z_attr = torch.empty(self.batch_size, cfg.N_ATTRIBUTES, H, W,).to(self.device)
        self.z_depth = torch.empty(self.batch_size, 1, H, W).to(self.device)
        self.z_pres = torch.empty(self.batch_size, 1, H, W).to(self.device)
        self.z_pres_prob = torch.empty(self.batch_size, 1, H, W).to(self.device)

        # s = torch.cuda.Stream()
        # # # Iterate through each grid cell and bounding boxes for that cell
        # with torch.cuda.stream(s):
        debug_tools.nan_hunter('Before Main Loop', input_data = x, edge_element=edge_element, backbone_feature=backbone_features)
        # test_context = torch.empty(self.batch_size, 55, H, W).to(self.device)

        for h, w in itertools.product(range(H), range(W)):
            # feature vec for each cell in the grid
            cell_feat = backbone_features[:, :, h, w]

            context = self._get_sequential_context(context_mat, h, w, edge_element)

            # --- z_where ---
            layer_inp = torch.cat((cell_feat, context), dim=-1)
            rep_input, passthru_features = self.box_network(layer_inp)
            box, normalized_box = self._build_box(rep_input, h, w)
            self.z_where[:, :, h, w, ] = normalized_box

            # --- z_what ---
            input_glimpses, attr_latent_var = self._encode_attr(x, normalized_box)
            attr_mean, attr_std = latent_to_mean_std(attr_latent_var)
            attr = self._sample_z(attr_mean, attr_std, 'attr', (h, w))
            self.z_attr[:, :, h, w, ] = attr

            # --- z_depth ---
            layer_inp = torch.cat([cell_feat, context, passthru_features, box, attr], dim=1)

            depth_latent, passthru_features = self.z_network(layer_inp)

            depth_mean, depth_std = latent_to_mean_std(depth_latent)
            depth_mean, depth_std = self._freeze_learning(depth_mean, depth_std)

            depth_logits = self._sample_z(depth_mean, depth_std, 'depth_logit', (h, w))
            depth = 4 * clamped_sigmoid(depth_logits)
            self.z_depth[:, :, h, w] = depth

            # --- z_presence ---
            layer_inp = torch.cat([cell_feat, context, passthru_features, box, attr, depth], dim=1)
            pres_logit = self.obj_network(layer_inp)
            obj_pres, obj_pres_prob = self._build_obj_pres(pres_logit)

            self.z_pres[:, :, h, w] = obj_pres
            self.z_pres_prob[:, :, h, w] = obj_pres_prob
            context_mat[(h, w)] = torch.cat((box, attr, depth, obj_pres), dim=-1)
            # test_context[..., h, w] = torch.cat((box, attr, depth), dim=-1)
            debug_tools.nan_hunter('Main Loop',
                                   grid_h=h,
                                   grid_w=w,
                                   context=context,
                                   normalized_box=normalized_box,
                                   attr=attr,
                                   depth=depth,
                                   presence=obj_pres
                                   )

        # Merge dist param, we have to use loop or autograd might not work
        for dist_name, dist_params in self.dist_param.items():
            means = self.dist_param[dist_name]['mean']
            sigmas = self.dist_param[dist_name]['sigma']
            self.dist[dist_name] = Normal(loc=means, scale=sigmas)


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
        input_chan = cfg.INPUT_IMAGE_SHAPE[0]
        n_inp_shape = obj_dim * obj_dim * input_chan # flattening the 14 x 14 x 3 image
        self.object_encoder = build_MLP(n_inp_shape, n_attr_out, hidden_layers=[256, 128])

        # object depth
        z_inp_shape = 4 + cfg.N_ATTRIBUTES + n_passthrough_features + self.context_dim + cfg.N_BACKBONE_FEATURES
        self.z_network = build_MLP(z_inp_shape, multiple_output=(2, n_passthrough_features)) # For training pass through features

        # object presence
        obj_inp_shape = z_inp_shape + 1 # incorporated z_network out dim (1 dim)
        self.obj_network = build_MLP(obj_inp_shape, 1)

        # object decoder
        input_chan = cfg.INPUT_IMAGE_SHAPE[0]
        decoded_dim = obj_dim * obj_dim * (input_chan+1) # [GrayScale + Alpha] or [RGB+A]
        self.object_decoder = build_MLP(cfg.N_ATTRIBUTES, decoded_dim, hidden_layers=[128, 256])

        self.attn = Self_Attn(55)

    def _build_box(self, latent_var, h, w):
        ''' Builds the bounding box from latent space z'''

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
        # hw ~ [0.0 , 1.0]
        height = clamped_sigmoid(height_logits)
        width = clamped_sigmoid(width_logits)
        max_hw = cfg.MAX_HW
        min_hw = cfg.MIN_HW
        assert max_hw > min_hw

        # current bounding box height & width ratio to anchor box
        height = float(max_hw - min_hw) * height + min_hw
        width = float(max_hw - min_hw) * width + min_hw

        box = torch.cat([cell_x, cell_y, width, height], dim=-1)

        # --- Compute image-normalized box parameters ---

        # box height and width normalized to image height and width
        anchor_box_dim = cfg.ANCHORBOX_SHAPE[0]
        _, image_height, image_width = cfg.INPUT_IMAGE_SHAPE
        # bounding box height & width relative to the whole image
        ys = height * anchor_box_dim / image_height
        xs = width * anchor_box_dim / image_width

        # box centre mapped with respect to full image
        yt = (self.pixels_per_cell[0] / image_height) * (cell_y + h)
        xt = (self.pixels_per_cell[1] / image_width) * (cell_x + w)

        normalized_box = torch.cat([xt, yt, xs, ys], dim=-1)

        if xs.min() < 0 or ys.min() < 0:
            print('oh shoot')

        return box, normalized_box

    def _sample_z(self, mean, var, name, cell_coord):
        '''
        Performs the sampling step in VAE and stores the distribution for KL computation
        :param mean:
        :param var:
        :param name: name of the distribution
        :return: sampled value
        '''
        dist = Normal(loc=mean, scale=var)

        if name not in self.dist_param.keys():
            _, H, W = self.feature_space_dim
            self.dist_param[name] = {}
            self.dist_param[name]['mean'] = torch.empty(self.batch_size, mean.shape[-1], H, W, ).to(self.device)
            self.dist_param[name]['sigma'] = torch.empty(self.batch_size, mean.shape[-1], H, W, ).to(self.device)
        x, y = cell_coord
        self.dist_param[name]['mean'][:, :, x, y] = mean
        self.dist_param[name]['sigma'][:, :, x, y] = var

        return dist.rsample()

class ConvSpair(SpairBase):

    def _build_networks(self):

        # backbone network
        self.backbone = Backbone(self.image_shape, cfg.N_BACKBONE_FEATURES)
        self.feature_space_dim = self.backbone.compute_output_shape()

        n_passthrough_features = cfg.N_PASSTHROUGH_FEATURES
        n_localization_latent = 8  # mean and var for (y, x, h, w)
        n_backbone_features = self.feature_space_dim[0]

        # bounding box
        self.z_where_net = LatentConv(n_backbone_features, n_localization_latent, additional_out_channels = n_passthrough_features)

        # object attribute
        n_attr_out = 2 * cfg.N_ATTRIBUTES
        obj_dim = cfg.OBJECT_SHAPE[0]
        input_chan = cfg.INPUT_IMAGE_SHAPE[0]
        n_inp_shape = obj_dim * obj_dim * input_chan # flattening the 14 x 14 x 3 image
        self.object_encoder = build_MLP(n_inp_shape, n_attr_out, hidden_layers = [256, 128])

        z_depth_in = 4 + cfg.N_ATTRIBUTES + n_passthrough_features + cfg.N_BACKBONE_FEATURES
        self.z_depth_net = LatentConv(z_depth_in, 2, additional_out_channels = n_passthrough_features)

        # object presence
        z_pres_in = z_depth_in + 1 # incorporated z_network out dim (1 dim)
        self.z_pres_net = LatentConv(z_pres_in, 1)

        # object decoder
        input_chan = cfg.INPUT_IMAGE_SHAPE[0]
        decoded_dim = obj_dim * obj_dim * (input_chan+1) # [GrayScale + Alpha] or [RGB+A]
        self.object_decoder = build_MLP(cfg.N_ATTRIBUTES, decoded_dim, hidden_layers=[128, 256])

        self.attn = Self_Attn(55)

    def _compute_latent_vars(self, x:torch.Tensor, backbone_features):
        _, H, W = self.feature_space_dim
        self.training_wheel = exponential_decay(**cfg.LATENT_VAR_TRAINING_WHEEL_PARAM)
        self.writer.add_scalar('training_wheel', self.training_wheel, self.global_step)

        # s = torch.cuda.Stream()
        # # # Iterate through each grid cell and bounding boxes for that cell
        # with torch.cuda.stream(s):
        debug_tools.nan_hunter('Before Main Loop', input_data=x, backbone_feature=backbone_features)
        # test_context = torch.empty(self.batch_size, 55, H, W).to(self.device)

        # --- z_where ---
        z_where, passthru_features = self.z_where_net(backbone_features)
        local_bbox, bbox = self._build_box(z_where)
        self.z_where = bbox

        # --- z_what ---
        bbox_hwc = to_H_W_C(bbox)
        # Repeat input H*W times so stn can process them at once
        batch_size, h, w, _ = bbox_hwc.shape
        x_expanded = x.unsqueeze(1).expand(-1, h * w, -1, -1, -1, )

        # [ Batch, H*W, ... ] -> [ Batch*H*W, ... ]
        bbox_bundle = bbox_hwc.contiguous().view(batch_size * h * w, -1 )
        x_bundle = x_expanded.contiguous().view(batch_size * h * w, self.image_shape[0], self.image_shape[1], self.image_shape[2])

        input_glimpses, z_what_bundle = self._encode_attr(x_bundle, bbox_bundle)

        # [ Batch*H*W, ... ] -> [ Batch,H,W, ... ]
        z_what = to_C_H_W(z_what_bundle.view(batch_size, h, w, -1))

        attr_mean, attr_std = latent_to_mean_std(z_what)
        attr = self._sample_z(attr_mean, attr_std, 'attr')
        self.z_attr = attr

        # --- z_depth ---
        layer_inp = torch.cat([backbone_features, passthru_features, local_bbox, attr], dim=1)

        depth_latent, passthru_features = self.z_depth_net(layer_inp)

        depth_mean, depth_std = latent_to_mean_std(depth_latent)
        depth_mean, depth_std = self._freeze_learning(depth_mean, depth_std)

        depth_logits = self._sample_z(depth_mean, depth_std, 'depth_logit')
        depth = 4 * clamped_sigmoid(depth_logits)
        self.z_depth = depth

        # --- z_presence ---
        layer_inp = torch.cat([backbone_features, passthru_features, local_bbox, attr, depth], dim=1)
        pres_logit = self.z_pres_net(layer_inp)
        obj_pres, obj_pres_prob = self._build_obj_pres(pres_logit)

        self.z_pres = obj_pres
        self.z_pres_prob = obj_pres_prob
        # test_context[..., h, w] = torch.cat((box, attr, depth), dim=-1)
        debug_tools.nan_hunter('Main Loop',
                               normalized_box=bbox,
                               attr=attr,
                               depth=depth,
                               presence=obj_pres
                               )

    def _build_box(self, z_where):
        ''' Builds the bounding box from latent space z'''

        mean, std = latent_to_mean_std(z_where)
        mean, std = self._freeze_learning(mean, std)

        #
        cy_mean, cx_mean, height_mean, width_mean = torch.chunk(mean, 4, dim=1)
        cy_std, cx_std, height_std, width_std = torch.chunk(std, 4, dim=1)

        cy_logits = self._sample_z(cy_mean, cy_std, 'cy_logit')
        cx_logits = self._sample_z(cx_mean, cx_std, 'cx_logit')
        height_logits = self._sample_z(height_mean, height_std, 'height_logit')
        width_logits = self._sample_z(width_mean, width_std, 'width_logit')

        cell_y = clamped_sigmoid(cy_logits)  # single digit
        cell_x = clamped_sigmoid(cx_logits)
        height = clamped_sigmoid(height_logits)
        width = clamped_sigmoid(width_logits)

        # --- map y,x, height, width relative to cell/anchor box ---

        # yx ~ [-0.5 , 1.5]
        max_yx = cfg.MAX_YX
        min_yx = cfg.MIN_YX
        assert max_yx > min_yx
        cell_y = float(max_yx - min_yx) * cell_y + min_yx
        cell_x = float(max_yx - min_yx) * cell_x + min_yx

        # hw ~ [0.0 , 1.0]
        max_hw = cfg.MAX_HW
        min_hw = cfg.MIN_HW
        assert max_hw > min_hw

        # current bounding box height & width ratio to anchor box
        height = float(max_hw - min_hw) * height + min_hw
        width = float(max_hw - min_hw) * width + min_hw

        box = torch.cat([cell_x, cell_y, width, height], dim=1)

        # --- Compute image-normalized box parameters ---

        # box height and width normalized to image height and width
        anchor_box_dim = cfg.ANCHORBOX_SHAPE[0]
        _, image_height, image_width = cfg.INPUT_IMAGE_SHAPE
        # bounding box height & width relative to the whole image
        ys = height * anchor_box_dim / image_height
        xs = width * anchor_box_dim / image_width

        # box centre mapped with respect to full image
        _, H, W = self.feature_space_dim
        h_offset = torch.arange(0, H, dtype=torch.float32,).unsqueeze(-1).expand_as(cell_y).to(RunManager.device)
        w_offset = torch.arange(0, W, dtype=torch.float32,).expand_as(cell_x).to(RunManager.device)
        yt = (self.pixels_per_cell[0] / image_height) * (cell_y + h_offset)
        xt = (self.pixels_per_cell[1] / image_width) * (cell_x + w_offset)

        normalized_box = torch.cat([xt, yt, xs, ys], dim=1)

        if xs.min() < 0 or ys.min() < 0:
            print('oh shoot')

        return box, normalized_box

    def _sample_z(self, mean, var, name):
        '''
        Performs the sampling step in VAE and stores the distribution for KL computation
        :param mean:
        :param var:
        :param name: name of the distribution
        :return: sampled value
        '''

        dist = Normal(loc=mean, scale=var)
        self.dist[name] = dist
        sampled = dist.rsample()
        return sampled

class ObjectConvEncoder(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        n_prev, h, w = input_size
        net = OrderedDict()

        # Builds internal layers except for the last layer
        for i, layer in enumerate(cfg.CONV_OBJECT_ENCODER_TOPOLOGY):
            layer['in_channels'] = n_prev

            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f #rename
            else:
                f = layer['out_channels']

            net['conv_%d' % i] = Conv2d(**layer)
            net['act_%d' % i] = ReLU()
            n_prev = f

        self.conv = Sequential(net)
        self.out = nn.Linear(123, output_size)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out_flat = conv_out.flatten(start_dim=1)
        return self.linear(conv_out_flat)

class ObjectConvDecoder(nn.Module):

    def __init__(self, input_size, output_channel):
        super().__init__()
        n_prev = input_size
        net = OrderedDict()
        decoder_topo = cfg.CONV_OBJECT_ENCODER_TOPOLOGY.reverse()
        # Builds internal layers except for the last layer
        for i, layer in enumerate(decoder_topo):
            layer['in_channels'] = n_prev

            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f #rename
            else:
                f = layer['out_channels']

            net['conv_transposed_%d' % i] = nn.ConvTranspose2d(**layer)

            net['act_%d' % i] = ReLU()
            n_prev = f
        net.pop()

        self.conv = Sequential(net)


    def forward(self, x):
        conv_out = torch.tensor(self.conv(x) )
        conv_out_flat = conv_out.flatten(start_dim=1)

        return self.linear(conv_out_flat)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        return out, attention



