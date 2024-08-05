from typing import Tuple, Optional
from enum import IntEnum, unique

import torch
from torchvision import ops

from trainingapi.data.structures.boxes import Boxes
from trainingapi.model.layers.box_iou_rotated import box_iou_rotated

@unique
class RotatedBoxMode(IntEnum):
    """
    Enum of different ways to represent a rotated box.
    """
    XYWHA_ABS = 0
    

class RotatedBoxes:
    def __init__(self, boxes, mode: RotatedBoxMode = RotatedBoxMode.XYWHA_ABS, bounds: Optional[Tuple[int, int]] = None):
        """
        Args:
            boxes (Tensor, Numpy, List of Lists[5]): a tensor of shape (N, 5) or (5, ), where the last dimension represents the box coordinates
                                                    in XYWHA_ABS only.
            mode (BoxMode): the format of the boxes             
            bounds (Tuple[int, int]): the height and width of the image to which the boxes belong
            
        Note:
            The box coordinates are always represented as float32 numbers.
            `bounds` is only used when converting boxes' representation to or from relative coordinates,
            so it can be None if no conversion will be called, i.e. convert_mode method is not called.
            If `bounds` is None, the conversion will raise an error if the mode is not XYXY_ABS or XYWH_ABS.
        """
        if not isinstance(boxes, torch.Tensor):
            # This shares memory with the original numpy array
            boxes = torch.as_tensor(boxes, dtype=torch.float32, device=torch.device("cpu"))
        
        else:
            boxes = boxes.to(dtype=torch.float32)
        
        if boxes.numel() == 0:
            # we dont end up creating a new tensor that does not depend on inputs and confuses jit
            boxes = torch.zeros((0, 5), dtype=torch.float32, device=boxes.device)
        
        assert boxes.size(-1) == 5, "boxes should be of shape Nx5"
        
        # Handles the case when boxes is an 1D tensor, i.e. a single box
        boxes = torch.atleast_2d(boxes) 
        
        self.boxes = boxes
        self.bounds = bounds
        self.mode = mode
    
    def __getitem__(self, indices) -> "RotatedBoxes":
        """
        Args: 
            indices (int, slice, LongTensor, BoolTensor): the indices to index
        Returns:
            Boxes: new Boxes that index the current tensor
        """
        if isinstance(indices, int):
            return RotatedBoxes(self.boxes[indices].view(1, -1), self.mode)
        
        return RotatedBoxes(self.boxes[indices], self.mode)

    def __repr__(self) -> str:
        return f"RotatedBoxes({self.boxes}, {self.mode})" 
    
    def __len__(self):
        return self.boxes.size(0)
    
    @torch.jit.unused
    def __iter__(self):
        yield from self.boxes
    
    def __eq__(self, other: "RotatedBoxes"):
        return torch.equal(self.boxes, other.convert_mode(self.mode).boxes)
        
    def device(self) -> torch.device:
        return self.boxes.device
    
    def clone(self) -> "RotatedBoxes":
        return RotatedBoxes(self.boxes.clone(), self.mode, self.bounds)

    def to(self, device: torch.device) -> "RotatedBoxes":
        return RotatedBoxes(self.boxes.to(device), self.mode, self.bounds)
    
    def clip_to_bounds(self, bounds: Tuple[int, int] = None) -> "RotatedBoxes":
        if bounds is None:
            assert self.bounds is not None, "bounds must be provided if self.bounds is not set"
            return self.clip_to_bounds(self.bounds)
        
        boxes = self.convert_mode(RotatedBoxMode.XYWHA_ABS).boxes
        boxes = ops.clip_boxes_to_image(boxes, bounds)
        return Boxes(boxes, RotatedBoxMode.XYWHA_ABS, self.bounds).convert_mode(self.mode) 
    
    def scale(self, scale_x: float, scale_y: float) -> "RotatedBoxes":
        self.boxes[:, 0] *= scale_x
        self.boxes[:, 1] *= scale_y
        theta = self.boxes[:, 4] * torch.pi / 180.0
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        scale_factor = torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)
        self.boxes[:, 2] *= scale_factor
        self.boxes[:, 3] *= scale_factor
        self.boxes[:, 4] = torch.atan2(scale_x * s, scale_y * c) * 180 / torch.pi
        return self
    
    def iou(self, other: "RotatedBoxes", angle_aware: bool = False) -> torch.Tensor:
        return box_iou_rotated(self.boxes, other.boxes, angle_aware=angle_aware)
    
    def area(self) -> torch.Tensor:
        area = torch.prod(self.boxes[:, 2:4], dim=1)
        return area

    def convert_mode(self, to_mode) -> "RotatedBoxes":
        raise NotImplementedError("convert_mode is not implemented for rotated boxes")
