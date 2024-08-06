from typing import List, Literal, Optional, Dict, Tuple

import torch
from torch import Tensor

def _resize_oboxes(oboxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    # Important: oboxes are (cx, cy, w, h, a)
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=oboxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=oboxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    cx, cy, w, h, a = oboxes.unbind(1)

    # Adjust angle
    # Converting angle from degrees to radians for calculation
    a_rad = torch.deg2rad(a)
    # Adjust the angle based on the change in aspect ratio
    sin_a = torch.sin(a_rad)
    cos_a = torch.cos(a_rad)
    
    a_adjusted_rad = torch.atan2(sin_a * ratio_height, cos_a * ratio_width)
    # Converting angle back to degrees
    a_adjusted = torch.rad2deg(a_adjusted_rad)
    a_adjusted = torch.where(a_adjusted < 0, a_adjusted + 360.0, a_adjusted)

    cx = cx * ratio_width
    cy = cy * ratio_height
    w = w * ratio_width
    h = h * ratio_height
    output = torch.stack((cx, cy, w, h, a_adjusted), dim=1)
    return output

def _flip_oboxes(oboxes: Tensor, new_size: List[int], direction: Literal['horizontal', 'vertical', 'diagonal']) -> Tensor:
    orig_shape = oboxes.shape
    oboxes = oboxes.reshape((-1, 5))
    flipped = oboxes.clone()
    height, width = new_size
    
    if direction == 'horizontal':
        flipped[:, 0] = width - oboxes[:, 0] - 1
        # Adjust angle for horizontal flip
        flipped[:, -1] = (180 - oboxes[:, -1]) % 360

    elif direction == 'vertical':
        flipped[:, 1] = height - oboxes[:, 1] - 1
        # Adjust angle for vertical flip
        flipped[:, -1] = (360 - oboxes[:, -1]) % 360

    elif direction == 'diagonal':
        flipped[:, 0] = width - oboxes[:, 0] - 1
        flipped[:, 1] = height - oboxes[:, 1] - 1
        # Adjust angle for diagonal flip
        # This combines both horizontal and vertical flips
        flipped[:, -1] = (180 - flipped[:, -1]) % 360
        flipped[:, -1] = (360 - flipped[:, -1]) % 360
    else:
        raise ValueError(f'Invalid flipping direction "{direction}"')
    
    return flipped.reshape(orig_shape)

def _resize_image(
    image: Tensor,
    self_min_size: float,
    self_max_size: float,
    target: Optional[Dict[str, Tensor]] = None,
    fixed_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    
    im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        scale_factor = scale.item()
        recompute_scale_factor = True

    image = torch.nn.functional.interpolate(
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    if target is None:
        return image, target
    
    return image, target