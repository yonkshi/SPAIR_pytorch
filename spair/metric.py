import torch

from spair import config as cfg

def mAP(z_where, z_pres, ground_truth_bbox, truth_bbox_digit_count):
    '''
    Computes the mean average precision (based on COCO dataset definition)

    WARNING: Assumes z_where and ground_truth_box both contain localization information in [X, Y, W, H] Format
    '''


    image_size = cfg.INPUT_IMAGE_SHAPE[-1]
    batch_size = z_where.shape[0]

    # clean up z_where to match ground_truth
    z_where *= image_size
    z_where = z_where.permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
    z_pres = z_pres.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)


    # turn height,width of bbox to max_x and max_y
    z_where[..., 2:] += z_where[..., :2]
    ground_truth_bbox[..., 2:] += ground_truth_bbox[..., :2]

    # masking away output unused bbox
    z_pres_rounded = torch.round(z_pres)
    z_where_masked = z_where * z_pres_rounded

    # mask away unused bbox in label
    # TODO Mask away unused bbox in label

    bbox_ious = batch_jaccard(z_where_masked, ground_truth_bbox)

    # choose the best output bbox to match label bbox
    bbox_iou = torch.max(bbox_ious, dim=1)[0] # [0] because max returns both max and argmax
    bbox_iou = bbox_iou.unsqueeze(-1).cpu()

    # Setup AP @ [0.1:0.1:0.9]
    # ap_scale = torch.arange(0.1, 1.0, 0.1)
    ap_scale = torch.tensor([0.0, 0.5], dtype=torch.float)
    scaled_iou = torch.clamp((bbox_iou - ap_scale) / (1 - ap_scale), min=0, max=1)

    # find the mean average precision (mAP)
    ap = scaled_iou.mean(dim=(-1))
    mean_ap = ap.sum(dim=-1, keepdim=True) / truth_bbox_digit_count.cpu() # normalize by num bboxes in label
    mean_ap = mean_ap.mean()
    return mean_ap

def object_count_accuracy(z_pres:torch.Tensor, truth_bbox_digit_count):

    batch_size = cfg.BATCH_SIZE
    z_pres = z_pres.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
    z_pres_count = z_pres.round().sum(dim = -2)

    count_accuracy = (truth_bbox_digit_count - z_pres_count).mean()
    return count_accuracy


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [Batch, A,1,2] -> [Batch, A,B,2]
    [B,2] -> [Batch, 1,B,2] -> [Batch, A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(1)
    B = box_b.size(1)
    batch = box_a.size(0)

    max_xy = torch.min(box_a[..., 2:].unsqueeze(2).expand(batch, A, B, 2),
                       box_b[..., 2:].unsqueeze(1).expand(batch, A, B, 2))
    min_xy = torch.max(box_a[..., :2].unsqueeze(2).expand(batch, A, B, 2),
                       box_b[..., :2].unsqueeze(1).expand(batch, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[..., 0] * inter[..., 1]


def batch_jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [Batch, num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [Batch, num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[..., 2]-box_a[..., 0]) *
              (box_a[..., 3]-box_a[..., 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[..., 2]-box_b[..., 0]) *
              (box_b[..., 3]-box_b[..., 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]