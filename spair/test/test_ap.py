
import numpy as np
import torch
import pickle
import os
def main():
    ap = AP(iou_threshold=[0.1])
    run_version = os.getenv('DEBUG_YONK_IMPLEMENTATION', False)
    ap_in = pickle.load(open("debug_AP.pickle", "rb"))
    ap(ap_in, debug_yonk= run_version)



# ------- begin yonk's IOU implementation ------\
def mAP2(z_where, z_pres, ground_truth_bbox, treshhold):
    '''
    Computes the mean average precision (based on COCO dataset definition)

    WARNING: Assumes z_where and ground_truth_box both contain localization information in [X, Y, W, H] Format
    '''

    if len(treshhold) > 1:
        return 0
    treshhold = treshhold[0]

    z_where = torch.tensor(z_where, dtype=torch.float).clone()
    z_pres = torch.tensor(z_pres, dtype=torch.float).unsqueeze(-1).clone()
    ground_truth_bbox = torch.tensor(ground_truth_bbox, dtype=torch.float).clone()

    z_where_top_bottom_left_right = z_where[...,[1,3,0,2]]
    pred_boxes = torch.cat([z_pres, z_where_top_bottom_left_right], dim=-1).numpy()
    gt_boxes = ground_truth_bbox[...,[1,3,0,2]].numpy()
    gt_count = [15] * ground_truth_bbox.shape[0]
    mAP = mAP_crawford(pred_boxes, gt_boxes, gt_count, iou_threshold=[0.1]) # TODO Add threshold

    return mAP
    # masking away output unused bbox
    z_pres_rounded = torch.round(z_pres)
    z_where_masked = z_where

    # mask away unused bbox in label
    # TODO Mask away unused bbox in label

    bbox_ious = batch_jaccard(z_where_masked, ground_truth_bbox)
    # choose the best output bbox to match label bbox
    best_bbox_iou, argbest_bbox_iout = torch.max(bbox_ious, dim=1)# [0] because max returns both max and argmax
    best_bbox_iou_thresh = torch.ceil(best_bbox_iou - treshhold)
    sorted_best_bbox_ious = sort_iou_by_confidence_across_batch(best_bbox_iou_thresh, argbest_bbox_iout, z_pres)
    sorted_best_bbox_ious = sorted_best_bbox_ious.cpu()

    # Compute AP

    cs = sorted_best_bbox_ious.cumsum(dim=0)
    precision = cs / (torch.arange(sorted_best_bbox_ious.shape[0], dtype=torch.float) + 1)
    n_positives_gt = sorted_best_bbox_ious.shape[0]
    recall_values = torch.linspace(0.0, 1.0, 11, dtype=torch.float)
    recall = cs / n_positives_gt

    _ap = []
    for r in recall_values:
        p = precision[recall >= r]
        _ap.append(0. if p.size()[0] == 0 else p.max())

    ap = torch.mean(torch.tensor(_ap))
    return ap, best_bbox_iou_thresh

def sort_iou_by_confidence_across_batch(best_bbox, best_bbox_idx, z_pres):
    ''' Sort the selected bboxes and sort it by confidence level (z_pres), returns only the IOU ratio of bbox. '''
    batch_size = z_pres.shape[0]
    batch_idx = torch.arange(batch_size).unsqueeze(-1)

    best_bbox_conf = z_pres[batch_idx, best_bbox_idx]

    conf_sorted, conf_sorted_idx = best_bbox_conf.flatten().sort(descending=True)
    flattend_best_bbox = best_bbox.flatten()
    sorted_best_bbox = flattend_best_bbox[conf_sorted_idx]
    return sorted_best_bbox


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

# ------- end Yonk's IOU implementation

def compute_iou(box, others):
    # box: y_min, y_max, x_min, x_max, area
    # others: (n_boxes, 5)
    top = np.maximum(box[0], others[:, 0])
    bottom = np.minimum(box[1], others[:, 1])
    left = np.maximum(box[2], others[:, 2])
    right = np.minimum(box[3], others[:, 3])

    overlap_height = np.maximum(0., bottom - top)
    overlap_width = np.maximum(0., right - left)
    overlap_area = overlap_height * overlap_width

    return overlap_area / (box[4] + others[:, 4] - overlap_area)
def mAP(pred_boxes, gt_boxes, z_where, z_pres, ground_truth_bbox, treshhold, n_classes,  recall_values=None, iou_threshold=None,):
    """ Calculate mean average precision on a dataset.

    Averages over:
        classes, recall_values, iou_threshold

    pred_boxes: [[class, conf, y_min, y_max, x_min, x_max] * n_boxes] * n_images
    gt_boxes: [[class, y_min, y_max, x_min, x_max] * n_boxes] * n_images

    """

    # if len(treshhold) > 1:
    #     return 0
    # treshhold = treshhold[0]
    #
    # z_where = torch.tensor(z_where, dtype=torch.float).clone()
    # z_pres = torch.tensor(z_pres, dtype=torch.float).clone()
    # ground_truth_bbox = torch.tensor(ground_truth_bbox, dtype=torch.float).clone()
    #
    # # masking away output unused bbox
    # z_pres_rounded = torch.round(z_pres)
    # z_where_masked = z_where
    #
    # # mask away unused bbox in label
    # # TODO Mask away unused bbox in label
    #
    # bbox_ious = batch_jaccard(z_where_masked, ground_truth_bbox)
    # # choose the best output bbox to match label bbox
    # # TODO Maybe sort predicted by confidence before selecting a best bbox
    # best_bbox_iou, argbest_bbox_iout = torch.max(bbox_ious, dim=2)# [0] because max returns both max and argmax
    # # TODO Remove duplicated bbox
    # best_bbox_iou_thresh = torch.ceil(best_bbox_iou - treshhold)
    # sorted_best_bbox_ious = sort_iou_by_confidence_across_batch(best_bbox_iou_thresh, argbest_bbox_iout, z_pres)
    # sorted_best_bbox_ious = sorted_best_bbox_ious.cpu()
    #
    # # Compute AP
    #
    # cs = sorted_best_bbox_ious.cumsum(dim=0)
    # precision = cs / (torch.arange(sorted_best_bbox_ious.shape[0], dtype=torch.float) + 1)
    # n_positives_gt = sorted_best_bbox_ious.shape[0]
    # recall_values = torch.linspace(0.0, 1.0, 11, dtype=torch.float)
    # recall = cs / n_positives_gt
    #
    # _ap = []
    # for r in recall_values:
    #     p = precision[recall >= r]
    #     _ap.append(0. if p.size()[0] == 0 else p.max())
    #
    # yonk_ap = torch.mean(torch.tensor(_ap))
    #
    # # TODO DEBUG STUFF
    #
    # bbox_ious_np = bbox_ious.numpy()
    #
    # # TODO END DEBUG STUFF


    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    if iou_threshold is None:
        iou_threshold = np.linspace(0.5, 0.95, 10)

    ap = []

    for c in range(n_classes):
        _ap = []
        for iou_thresh in iou_threshold:
            predicted_list = []  # Each element is of the form (confidence, ground-truth (0 or 1))
            n_positives_gt = 0

            debug_selected_gt_bbox = np.zeros((32, 15), dtype=np.int)
            debug_iou = np.zeros((32,49,15), dtype=np.float32)
            batch_idx = 0
            for pred, gt in zip(pred_boxes, gt_boxes): # FIXME Zip along batch dim
                # Within a single image

                # Sort by decreasing confidence within current class. High to low
                pred_c = [b for cls, *b in pred if cls == c]
                # pred_c = sorted([b for cls, *b in pred if cls == c], key=lambda k: -k[0])
                area = [(ymax - ymin) * (xmax - xmin) for _, ymin, ymax, xmin, xmax in pred_c]
                pred_c = [(*b, a) for b, a in zip(pred_c, area)]         # pred_c = [conf, x, y, h, w, area]

                gt_c = [b for cls, *b in gt if cls == c]
                n_positives_gt += len(gt_c)

                if not gt_c:
                    predicted_list.extend((conf, 0) for conf, *_ in pred_c)
                    continue

                gt_c = np.array(gt_c)
                gt_c_area = (gt_c[:, 1] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 2])
                gt_c = np.concatenate([gt_c, gt_c_area[..., None]], axis=1)

                used = [0] * len(gt_c)
                bbox_idx = 0
                for conf, *box in pred_c:
                    iou = compute_iou(box, gt_c)
                    best_idx = np.argmax(iou)
                    best_iou = iou[best_idx]
                    debug_iou[batch_idx, bbox_idx] = iou

                    if best_iou > iou_thresh and not used[best_idx]:
                        predicted_list.append((conf, 1.))
                        used[best_idx] = 1
                        debug_selected_gt_bbox[batch_idx, best_idx] = bbox_idx
                    else:
                        predicted_list.append((conf, 0.))
                    bbox_idx +=1
                batch_idx += 1
            if not predicted_list:
                ap.append(0.0)
                continue

            # Sort predictions by decreasing confidence.
            predicted_list_sorted = np.array(sorted(predicted_list, key=lambda k: -k[0]), dtype=np.float32)

            # Compute AP
            cs = np.cumsum(predicted_list_sorted[:, 1])
            precision = cs / (np.arange(predicted_list_sorted.shape[0]) + 1)
            recall = cs / n_positives_gt

            for r in recall_values:
                p = precision[recall >= r]
                _ap.append(0. if p.size == 0 else p.max())

        ap.append(np.mean(_ap) if _ap else 0.0)
        dbg = np.array(predicted_list)[:, 1]
    return np.mean(ap), dbg


def mAP_crawford(pred_boxes, gt_boxes, gt_boxes_count, recall_values=None, iou_threshold=None):
    """ Calculate mean average precision on a dataset.

    Averages over:
        classes, recall_values, iou_threshold

    pred_boxes: [[class, conf, y_min, y_max, x_min, x_max] * n_boxes] * n_images
    gt_boxes: [[class, y_min, y_max, x_min, x_max] * n_boxes] * n_images

    """
    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    _ap = []
    for iou_thresh in iou_threshold:
        predicted_list = []  # Each element is of the form (confidence, ground-truth (0 or 1))
        n_positives_gt = 0

        for pred, gt, gt_count in zip(pred_boxes, gt_boxes, gt_boxes_count):
            # Within a single image

            # Sort by decreasing confidence within current class.
            pred_c = sorted(pred, key=lambda k: -k[0])
            area = [(ymax - ymin) * (xmax - xmin) for _, ymin, ymax, xmin, xmax in pred_c]
            pred_c = [(*b, a) for b, a in zip(pred_c, area)]

            gt_c = gt
            n_positives_gt += gt_count

            gt_c = np.array(gt_c)
            gt_c_area = (gt_c[:, 1] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 2])
            gt_c = np.concatenate([gt_c, gt_c_area[..., None]], axis=1)

            used = [0] * gt_count

            for conf, *box in pred_c:
                iou = compute_iou(box, gt_c)
                best_idx = np.argmax(iou)
                best_iou = iou[best_idx]

                if best_iou > iou_thresh and not used[best_idx]:
                    predicted_list.append((conf, 1.))
                    used[best_idx] = 1
                else:
                    predicted_list.append((conf, 0.))

        if not predicted_list:
            _ap.append(0.0)
            continue

        # Sort predictions by decreasing confidence.
        predicted_list = np.array(sorted(predicted_list, key=lambda k: -k[0]), dtype=np.float32)

        # Compute AP
        cs = np.cumsum(predicted_list[:, 1])
        precision = cs / (np.arange(predicted_list.shape[0]) + 1)
        recall = cs / n_positives_gt

        for r in recall_values:
            p = precision[recall >= r]
            _ap.append(0. if p.size == 0 else p.max())

    return np.mean(_ap)

class AP:
    keys_accessed = "normalized_box obj annotations n_annotations"

    def __init__(self, iou_threshold=None, start_frame=0, end_frame=np.inf, is_training=False):
        if iou_threshold is not None:
            try:
                iou_threshold = list(iou_threshold)
            except (TypeError, ValueError):
                iou_threshold = [float(iou_threshold)]
        self.iou_threshold = iou_threshold # FIXME [0.1:0.1:0.9]

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.is_training = is_training

    def get_feed_dict(self, updater):
        if self.is_training:
            return {updater.network.is_training: True}
        else:
            return {}

    def _process_data(self, tensors):
        anchorbox = (48, 48)
        image_size = (84, 84 )
        nb = np.split(tensors['normalized_box'], 4, axis=-1)
        top, left, height, width = coords_to_pixel_space(
            *nb, (image_size[0], image_size[1]),
            anchorbox, top_left=True)

        obj = tensors['obj']
        batch_size = obj.shape[0]
        n_frames = 0

        annotations = tensors["annotations"]
        n_annotations = tensors["n_annotations"]

        if n_frames > 0:
            n_objects = np.prod(obj.shape[2:-1])
            n_frames = obj.shape[1]
        else:
            n_objects = np.prod(obj.shape[1:-1])
            annotations = annotations.reshape(batch_size, 1, *annotations.shape[1:])
            n_frames = 1

        shape = (batch_size, n_frames, int(n_objects))

        n_digits = n_objects * np.ones((batch_size, n_frames), dtype=np.int32)

        obj = obj.reshape(*shape)
        top = top.reshape(*shape)
        left = left.reshape(*shape)
        height = height.reshape(*shape)
        width = width.reshape(*shape)

        return obj, n_digits, top, left, height, width, annotations, n_annotations

    def __call__(self, tensors, debug_yonk=False):
        obj, n_digits, top, left, height, width, annotations, n_annotations = self._process_data(tensors)

        bottom = top + height
        right = left + width

        batch_size, n_frames = n_digits.shape[:2]

        ground_truth_boxes = []
        predicted_boxes = []

        for b in range(batch_size):
            for f in range(self.start_frame, min(self.end_frame, n_frames)):
                _ground_truth_boxes = []
                for debug_var in zip(annotations[b, f], range(n_annotations[b])):
                    (valid, _, _, *bbox), _  = debug_var
                    if valid > 0.5:
                        _ground_truth_boxes.append((0, *bbox))

                ground_truth_boxes.append(_ground_truth_boxes)

                _predicted_boxes = []
                for j in range(int(n_digits[b, f])):
                    o = obj[b, f, j]

                    if o > 0.0:
                        _predicted_boxes.append(
                            [0, o,
                             top[b, f, j],
                             bottom[b, f, j],
                             left[b, f, j],
                             right[b, f, j]])

                predicted_boxes.append(_predicted_boxes)
        np_predicted = np.array(predicted_boxes)
        z_where_tblr = np_predicted[...,2:]
        z_pres = np_predicted[...,1]
        # from top-bottom-left-right to x, y, w, h
        z_where = z_where_tblr[..., [2,0,3,1]]

        np_ground_truth_tblr = np.array(ground_truth_boxes)[...,1:]
        # from top-bottom-left-right to x, y, w, h
        np_ground_truth = np_ground_truth_tblr[..., [2,0,3,1]]

        map_val = mAP2(z_where, z_pres, np_ground_truth, self.iou_threshold)
        map_val, predicted_list = mAP(
        predicted_boxes, ground_truth_boxes,
            z_where, z_pres, np_ground_truth, self.iou_threshold, # Yonk stuff
            n_classes=1,
        iou_threshold=self.iou_threshold)

        return map_val

def coords_to_pixel_space(y, x, h, w, image_shape, anchor_box, top_left):
    h = h * anchor_box[0]
    w = w * anchor_box[1]

    y = y * anchor_box[0] - 0.5
    x = x * anchor_box[1] - 0.5

    if top_left:
        y -= h / 2
        x -= w / 2

    return y, x, h, w

if __name__ == '__main__':
    main()