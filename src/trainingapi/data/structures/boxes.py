from typing import Tuple, Optional
from enum import IntEnum, unique

import torch
from torchvision import ops

@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """
    XYXY_ABS = 0
    XYWH_ABS = 1
    CXCYWH_ABS = 2
    XYXY_REL = 3
    XYWH_REL = 4
    CXCYWH_REL = 5
    
class Boxes:
    def __init__(self, boxes, mode: BoxMode, bounds: Optional[Tuple[int, int]] = None):
        """
        Args:
            boxes (Tensor, Numpy, List of Lists[4]): a tensor of shape (N, 4) or (4, ), where the last dimension represents the box coordinates
                                                    in XYXY_ABS, XYWH_ABS, CXCYWH_ABS, XYXY_REL, XYWH_REL, or CXCYWH_REL format.
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
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=boxes.device)
        
        assert boxes.size(-1) == 4, "boxes should be of shape Nx4"
        
        # Handles the case when boxes is an 1D tensor, i.e. a single box
        boxes = torch.atleast_2d(boxes) 
        
        self.boxes = boxes
        self.bounds = bounds
        self.mode = mode
    
    def __getitem__(self, indices) -> "Boxes":
        """
        Args: 
            indices (int, slice, LongTensor, BoolTensor): the indices to index
        Returns:
            Boxes: new Boxes that index the current tensor
        """
        if isinstance(indices, int):
            return Boxes(self.boxes[indices].view(1, -1), self.mode)
        
        return Boxes(self.boxes[indices], self.mode)

    def __repr__(self) -> str:
        return f"Boxes({self.boxes}, {self.mode})" 
    
    def __len__(self):
        return self.boxes.size(0)
    
    @torch.jit.unused
    def __iter__(self):
        yield from self.boxes
    
    def __eq__(self, other: "Boxes"):
        return torch.equal(self.boxes, other.convert_mode(self.mode).boxes)
        
    def device(self) -> torch.device:
        return self.boxes.device
    
    def clone(self) -> "Boxes":
        return Boxes(self.boxes.clone(), self.mode, self.bounds)

    def to(self, device: torch.device) -> "Boxes":
        return Boxes(self.boxes.to(device), self.mode, self.bounds)
    
    def clip_to_bounds(self, bounds: Tuple[int, int]) -> "Boxes":
        if bounds is None:
            assert self.bounds is not None, "bounds must be provided if self.bounds is not set"
            return self.clip_to_bounds(self.bounds)
        
        boxes = self.convert_mode(BoxMode.XYXY_ABS).boxes
        boxes = ops.clip_boxes_to_image(boxes, bounds)
        return Boxes(boxes, BoxMode.XYXY_ABS, self.bounds).convert_mode(self.mode) 
    
    def scale(self, scale_x: float, scale_y: float) -> "Boxes":
        self.boxes[:, 0::2] *= scale_x
        self.boxes[:, 1::2] *= scale_y
        return self
    
    def iou(self, other: "Boxes") -> torch.Tensor:
        return ops.box_iou(self.convert_mode(BoxMode.XYXY_ABS).boxes, other.convert_mode(BoxMode.XYXY_ABS).boxes)
    
    def area(self) -> torch.Tensor:
        if self.mode == BoxMode.XYXY_ABS:
            area = torch.prod(self.boxes[:, 2:] - self.boxes[:, :2], dim=1)
            
        elif self.mode == BoxMode.XYWH_ABS:
            area = torch.prod(self.boxes[:, 2:], dim=1)
            
        elif self.mode == BoxMode.CXCYWH_ABS:
            area = torch.prod(self.boxes[:, 2:], dim=1)
            
        elif self.mode == BoxMode.XYXY_REL:
            assert self.bounds is not None, "Cannot compute area for relative boxes without bounds"
            area = torch.prod(self.boxes[:, 2:] - self.boxes[:, :2], dim=1) * self.bounds[0] * self.bounds[1]
            
        elif self.mode == BoxMode.XYWH_REL:
            assert self.bounds is not None, "Cannot compute area for relative boxes without bounds"
            area = torch.prod(self.boxes[:, 2:], dim=1) * self.bounds[0] * self.bounds[1]
            
        elif self.mode == BoxMode.CXCYWH_REL:
            assert self.bounds is not None, "Cannot compute area for relative boxes without bounds"
            area = torch.prod(self.boxes[:, 2:], dim=1) * self.bounds[0] * self.bounds[1]
            
        return area

    def convert_mode(self, to_mode: BoxMode) -> "Boxes":
        if to_mode == self.mode:
            return self.clone()

        boxes = self.boxes.clone()
        if self.mode in (BoxMode.XYXY_REL, BoxMode.XYWH_REL, BoxMode.CXCYWH_REL) or to_mode in (BoxMode.XYXY_REL, BoxMode.XYWH_REL, BoxMode.CXCYWH_REL):
            if self.bounds is None:
                raise ValueError("Cannot convert the boxes without bounds")

        scale = torch.tensor([self.bounds[1], self.bounds[0], self.bounds[1], self.bounds[0]], device=boxes.device)

        if self.mode == BoxMode.XYXY_ABS:
            match to_mode:
                case BoxMode.XYXY_REL:
                    boxes /= scale

                case BoxMode.XYWH_REL:
                    boxes[:, 2:] -= boxes[:, :2]
                    boxes /= scale

                case BoxMode.XYWH_ABS:
                    boxes[:, 2:] -= boxes[:, :2]

                case BoxMode.CXCYWH_ABS:
                    boxes[:, :2], boxes[:, 2:]  = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]

                case BoxMode.CXCYWH_REL:
                    boxes[:, :2], boxes[:, 2:] = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]
                    boxes /= scale

                case _:
                    raise NotImplementedError(f"Unimplemented BoxMode: {to_mode}")

        elif self.mode == BoxMode.XYWH_ABS:
            match to_mode:
                case BoxMode.XYXY_REL:
                    boxes[:, 2:] += boxes[:, :2]
                    boxes /= scale

                case BoxMode.XYWH_REL:
                    boxes /= scale

                case BoxMode.XYXY_ABS:
                    boxes[:, 2:] += boxes[:, :2]

                case BoxMode.CXCYWH_ABS:
                    boxes[:, :2] += boxes[:, 2:] / 2

                case BoxMode.CXCYWH_REL:
                    boxes[:, :2] += boxes[:, 2:] / 2
                    boxes /= scale

                case _:
                    raise NotImplementedError(f"Unimplemented BoxMode: {to_mode}")

        elif self.mode == BoxMode.CXCYWH_ABS:
            match to_mode:
                case BoxMode.XYXY_ABS:
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes[:, 2:] += boxes[:, :2]

                case BoxMode.XYWH_ABS:
                    boxes[:, :2] -= boxes[:, 2:] / 2

                case BoxMode.XYXY_REL:
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes[:, 2:] += boxes[:, :2]
                    boxes /= scale

                case BoxMode.XYWH_REL:
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes /= scale

                case BoxMode.CXCYWH_REL:
                    boxes /= scale

                case _:
                    raise NotImplementedError(f"Unimplemented BoxMode: {to_mode}")

        elif self.mode == BoxMode.XYXY_REL:
            match to_mode:
                case BoxMode.XYXY_ABS:
                    boxes *= scale

                case BoxMode.XYWH_REL:
                    boxes[:, 2:] -= boxes[:, :2]

                case BoxMode.XYWH_ABS:
                    boxes *= scale
                    boxes[:, 2:] -= boxes[:, :2]

                case BoxMode.CXCYWH_ABS:
                    boxes *= scale
                    boxes[:, :2], boxes[:, 2:] = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]

                case BoxMode.CXCYWH_REL:
                    boxes[:, :2], boxes[:, 2:] = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]

                case _:
                    raise NotImplementedError(f"Unimplemented BoxMode: {to_mode}")

        elif self.mode == BoxMode.XYWH_REL:
            match to_mode:
                case BoxMode.XYXY_ABS:
                    boxes *= scale
                    boxes[:, 2:] += boxes[:, :2]

                case BoxMode.XYWH_ABS:
                    boxes *= scale

                case BoxMode.XYXY_REL:
                    boxes[:, 2:] += boxes[:, :2]

                case BoxMode.CXCYWH_ABS:
                    boxes *= scale
                    boxes[:, :2] += boxes[:, 2:] / 2

                case BoxMode.CXCYWH_REL:
                    boxes[:, :2] += boxes[:, 2:] / 2

                case _:
                    raise NotImplementedError(f"Unimplemented BoxMode: {to_mode}")

        elif self.mode == BoxMode.CXCYWH_REL:
            match to_mode:
                case BoxMode.XYXY_ABS:
                    boxes *= scale
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes[:, 2:] += boxes[:, :2]

                case BoxMode.XYWH_ABS:
                    boxes *= scale
                    boxes[:, :2] -= boxes[:, 2:] / 2

                case BoxMode.XYXY_REL:
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes[:, 2:] += boxes[:, :2]

                case BoxMode.XYWH_REL:
                    boxes[:, :2] -= boxes[:, 2:] / 2

                case BoxMode.CXCYWH_ABS:
                    boxes *= scale

                case _:
                    raise NotImplementedError(f"Unimplemented BoxMode: {to_mode}")

        return Boxes(boxes, to_mode, self.bounds)
