import torch 
try:
    import trainingapi._C 

except ImportError:
    raise ImportError("Cannot find trainingapi._C. Have you run `pip install -e .`?")

from .anchors_rotated import RotatedAnchorGenerator
from .box_ops_rotated import XYWHA_XYWHA_BoxCoder, remove_small_rotated_boxes
from .box_iou_rotated import box_iou_rotated
from .nms_rotated import batched_nms_rotated
from .roi_align_rotated import MultiScaleRotatedRoIAlign