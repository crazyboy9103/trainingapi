import torch
from trainingapi._C import box_iou_rotated as _C_box_iou_rotated

def box_iou_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor, aligned: bool = True) -> torch.Tensor:
    """ Rotated box IoU. mode_flag and aligned are kept for compatibility with mmrotate implementation.
    Args:
        boxes1, boxes2 (Tensor[N, 5]): boxes in ``(cx, cy, w, h, a)`` format
        mode_flag (int): 0: standard IOU (Union is a+b-a&b), 1: IOU (Union is a)
        aligned (bool): in principle, aligned=True performs better, but the difference is not significant
    
    Returns:
        Tensor[N, N]: the NxN matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    return _C_box_iou_rotated(boxes1, boxes2, aligned)
