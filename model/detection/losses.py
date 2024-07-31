from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

def rotated_faster_rcnn_loss(
    class_logits: Tensor, 
    obox_regression: Tensor, 
    labels: List[Tensor],
    obox_regression_targets: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes the loss for Rotated Faster R-CNN.

    Args:
        class_logits (Tensor) : N x C
        obox_regression (Tensor) : (N x C x 5)
        labels (list[Tensor])
        obox_regression_targets (Tensor)
        
    Returns:
        classification_loss (Tensor)
        obox_loss (Tensor)
    """
   
    N, num_classes = class_logits.shape
    
    labels = torch.cat(labels, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
        
    pos_inds = (labels > 0).nonzero().squeeze()
    labels_pos = labels[pos_inds]
    
    obox_regression_targets = torch.cat(obox_regression_targets, dim=0)
    obox_regression = obox_regression.reshape(N, obox_regression.size(-1) // 6, 6) # N x C x 5
    
    preds = obox_regression[pos_inds, labels_pos]
    targets = obox_regression_targets[pos_inds]
    
    preds_coords = preds[:, :4]
    preds_angles = preds[:, 4:]

    targets_coords = targets[:, :4]
    targets_angles = targets[:, 4:]

    obox_loss = F.smooth_l1_loss(
        preds_coords, 
        targets_coords,
        beta=1.0 / 9,
        reduction="sum",
    ) / labels.numel()

    angle_loss = F.smooth_l1_loss(
        preds_angles, 
        targets_angles,
        beta=1.0,
        reduction="sum",
    ) / labels.numel()
    return classification_loss, obox_loss, angle_loss