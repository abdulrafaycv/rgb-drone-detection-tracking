"""
OC-SORT association utilities.
Source: https://github.com/noahcao/OC_SORT (MIT License)
Indentation cleaned for standalone use.
"""
import numpy as np


def iou_batch(bboxes1, bboxes2):
    """Computes IOU between two sets of bboxes in [x1,y1,x2,y2] format."""
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o


def giou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
             + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    area_enclose = (xxc2 - xxc1) * (yyc2 - yyc1)
    giou = iou - (area_enclose - union) / (area_enclose + 1e-7)
    return (giou + 1.) / 2.0


def diou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
             + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union
    cx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.
    cy1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.
    cx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.
    cy2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / (outer_diag + 1e-7)
    return (diou + 1) / 2.0


def ciou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
             + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union
    cx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.
    cy1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.
    cx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.
    cy2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.
    inner_diag = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1] + 1.
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1] + 1.
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou
    alpha = v / (S + v + 1e-7)
    ciou = iou - inner_diag / (outer_diag + 1e-7) - alpha * v
    return (ciou + 1) / 2.0


def ct_dist(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    cx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.
    cy1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.
    cx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.
    cy2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.
    ct_dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    ct_dist_val = np.sqrt(ct_dist2)
    max_val = ct_dist_val.max()
    if max_val == 0:
        return ct_dist_val
    ct_dist_val = ct_dist_val / max_val
    return ct_dist_val.max() - ct_dist_val


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1 = (dets[:, 0] + dets[:, 2]) / 2.
    CY1 = (dets[:, 1] + dets[:, 3]) / 2.
    CX2 = (tracks[:, 0] + tracks[:, 2]) / 2.
    CY2 = (tracks[:, 1] + tracks[:, 3]) / 2.
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def _filter_matches(matched_indices, iou_matrix, iou_threshold, num_dets, num_trks):
    if matched_indices.shape[0] > 0:
        unmatched_dets = np.setdiff1d(np.arange(num_dets), matched_indices[:, 0])
        unmatched_trks = np.setdiff1d(np.arange(num_trks), matched_indices[:, 1])
        iou_vals = iou_matrix[matched_indices[:, 0], matched_indices[:, 1]]
        low_iou_mask = iou_vals < iou_threshold
        unmatched_dets = np.concatenate([unmatched_dets, matched_indices[low_iou_mask, 0]])
        unmatched_trks = np.concatenate([unmatched_trks, matched_indices[low_iou_mask, 1]])
        matches = matched_indices[~low_iou_mask]
    else:
        unmatched_dets = np.arange(num_dets)
        unmatched_trks = np.arange(num_trks)
        matches = np.empty((0, 2), dtype=int)
    return matches, unmatched_dets.astype(int), unmatched_trks.astype(int)


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y = velocities[:, 0][:, np.newaxis]
    inertia_X = velocities[:, 1][:, np.newaxis]
    diff_angle_cos = np.clip(inertia_X * X + inertia_Y * Y, -1, 1)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi
    valid_mask = (previous_obs[:, 4] >= 0).astype(float)[:, np.newaxis]
    iou_matrix = iou_batch(detections, trackers)
    scores = detections[:, -1][:, np.newaxis]
    angle_diff_cost = (valid_mask * diff_angle * vdc_weight).T * scores
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))
    return _filter_matches(matched_indices, iou_matrix, iou_threshold, len(detections), len(trackers))
