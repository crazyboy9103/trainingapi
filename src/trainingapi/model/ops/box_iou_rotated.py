import torch
from trainingapi._C import box_iou_rotated as _C_box_iou_rotated

def box_iou_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor, angle_aware: bool = True) -> torch.Tensor:
    """ Rotated box IoU.
    Args:
        boxes1, boxes2 (Tensor[N, 5]): boxes in ``(cx, cy, w, h, a)`` format
        angle_aware: arIoU introduced in https://arxiv.org/pdf/1711.09405.pdf
                     if angle_aware=True, IoU *= max(0, cos(box1_a - box2_a)), 
                     i.e. modulates IoU by a factor <= 1 and suppresses IoU when angle difference is within (90, 270)
    Returns:
        Tensor[N, N]: the NxN matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    # print(boxes1.dtype, boxes2.dtype)
    return _C_box_iou_rotated(boxes1.contiguous(), boxes2.contiguous(), angle_aware)