import torch 
try:
    import detectron2._C 

except ImportError:
    raise ImportError("Cannot find detectron2._C. Have you run `python setup.py install`?")

from .anchors_rotated import RotatedAnchorGenerator
from .box_ops_rotated import XYWHA_XYWHA_BoxCoder, remove_small_rotated_boxes
from .box_iou_rotated import box_iou_rotated
from .nms_rotated import batched_nms_rotated
from .roi_align_rotated import MultiScaleRotatedRoIAlign