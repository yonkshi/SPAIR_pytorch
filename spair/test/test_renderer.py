from spair import config as cfg
from spair.modules import *
import numpy as np
import torch
from torch import nn


def test():
    batch = 2
    grid_size = 11
    obj_size = 28
    channels = 2
    z_depth = torch.ones([batch, 1, grid_size, grid_size])
    z_pres = torch.ones([batch, 1, grid_size, grid_size])

    hw = 1 / 11
    where = torch.tensor([0., 0., hw, hw,])[None, :, None, None]
    z_where = where.repeat([batch, 1, grid_size, grid_size])
    offsets = torch.linspace(0, 1, steps=12)[:-1]
    z_where[0, 0, :, : ] += offsets
    z_where[0, 1, :, :] += offsets[:, None]

    npwhere = z_where[0, :2, :, :].numpy()

    objects = torch.ones([batch, grid_size, grid_size, obj_size, obj_size, 1]) * -1000
    objects[0, 0, 8, 14, :, 0] += 2000 # row 4, col 5

    # alpha
    alpha = torch.ones_like(objects) * 1000

    object_logits = torch.cat([objects, alpha], dim=-1)
    # flattening for testing
    object_logits = object_logits.view(batch * grid_size * grid_size, obj_size * obj_size * 2)
    _render(object_logits, z_where, z_depth, z_pres)
    print('hello')
    pass


def _render( object_logits, z_where, z_depth, z_pres):
    '''
    decoder + renderer function. combines the latent vars & bbox, feed into the decoder for each object and then
    :param z_attr:
    :param z_where:
    :param z_depth:
    :param z_pres:
    :return:
    '''
    H, W = 11, 11
    px = cfg.OBJECT_SHAPE[0]

    # ---- Now entering B x H x W x C realm, because we needed to merge B*H*W ----

    # Permute then flatten latent vars from [Batch, ?, H, W, ?] to [Batch * H * W, ?] so we can feed into STN
    z_where = to_H_W_C(z_where).contiguous().view(-1, 4)

    # flattening z depth and z presence for element-wise broadcasted multiplication later
    z_depth = z_depth.view(-1, 1, 1)
    z_pres = z_pres.view(-1, 1, 1)
    # object_decoder_in = to_H_W_C(z_attr).contiguous().view(-1, cfg.N_ATTRIBUTES)

    # MLP to generate image
    # object_logits = self.object_decoder(object_decoder_in)
    input_chan_w_alpha = cfg.INPUT_IMAGE_SHAPE[0] + 1
    object_logits = object_logits.view(-1, px, px,
                                       input_chan_w_alpha)  # [Batch * n_cells, pixels_w, pixels_h, channels]

    # object_logits scale + bias mask
    object_logits[:, :, :, :-1] *= cfg.OBJ_LOGIT_SCALE  # [B, 14, 14, 4] * [4]
    object_logits[:, :, :, -1] *= cfg.ALPHA_LOGIT_SCALE
    object_logits[:, :, :, -1] += cfg.ALPHA_LOGIT_BIAS

    objects = clamped_sigmoid(object_logits, use_analytical=True)
    objects = objects.view(-1, px, px, input_chan_w_alpha)

    # incorporate presence in alpha channel
    objects[:, :, :, -1] *= z_pres.expand_as(objects[:, :, :, -1])

    # importance manipulates how gradients scales, but does not nessasarily
    importance = objects[:, :, :, -1] * z_depth.expand_as(objects[:, :, :, -1])
    importance = torch.clamp(importance, min=0.01)

    # Merge importance to objects:
    importance = importance[..., None]  # add a trailing dim for concatnation
    objects = torch.cat([objects, importance], dim=-1)  # attach importance to RGBA, 5 channels to total

    debug_tools.plot_prerender_components(objects, z_pres, z_depth, z_where, None, 0)
    # ---- exiting B x H x W x C realm .... ----

    objects_ = to_C_H_W(objects)

    img_c, img_h, img_w, = (1, 128, 128)
    n_obj = H * W  # max number of objects in a grid
    transformed_imgs = stn(objects_, z_where, [img_h, img_w], torch.device('cpu'), inverse=True)
    transformed_objects = transformed_imgs.contiguous().view(-1, n_obj, img_c + 2, img_h, img_w)
    # incorporate alpha
    # FIXME The original implement doesn't seem to be calculating alpha correctly.
    #  If multiple objects overlap one pixel, alpha is only computed against background
    # TODO assume background is black. Will learn to construct different background in the future

    # TODO we can potentially compute alpha and importance prior to stn, it will be much faster

    input_chan = cfg.INPUT_IMAGE_SHAPE[0]
    color_channels = transformed_objects[:, :, :input_chan, :, :]
    alpha = transformed_objects[:, :, input_chan:input_chan + 1, :, :]  # keep the empty dim
    importance = transformed_objects[:, :, input_chan + 1:input_chan + 2, :, :]

    img = alpha.expand_as(color_channels) * color_channels

    # normalize importance
    importance = importance / importance.sum(dim=1, keepdim=True)
    importance = importance.expand_as(img)
    # scale gradient
    weighted_grads_image = img * importance  # + (1 - importance) * img.detach()

    output_image = weighted_grads_image.sum(dim=1)  # sum up along n_obj per image
    debug_tools.plot_torch_image_in_pyplot(output_image)
    return output_image

if __name__ == '__main__':
    test()