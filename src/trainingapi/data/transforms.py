from typing import List, Optional, Tuple, Any, Dict
import math

import torch
from torch import nn
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models.detection.image_list import ImageList

from trainingapi.data.preprocess import _resize_oboxes, _flip_oboxes, _resize_image

class RotatedFasterRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a RotatedFasterRCNNTransform
    model.

    The transformations it performs are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        image_mean: List[float],
        image_std: List[float],
        size_divisible: int = 32,
        fixed_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size
        self._skip_resize = kwargs.pop("_skip_resize", False)
        self._skip_flip = kwargs.pop("_skip_flip", False)
        self._skip_image_transform = kwargs.pop("_skip_image_transform", False)

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        
        image_transform = self.get_image_transform(self.training)
        for i, image in enumerate(images):
            target = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            
            if not self._skip_image_transform and image_transform:
                image = image_transform(image)
           
            image, target = self.resize(image, target)
            
            if not self._skip_flip and self.training:
                image, target = self.flip_bboxes(image, target)
            
            image = self.normalize(image)
            
            images[i] = image
            if targets is not None and target is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image: Tensor) -> Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops, so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
        return k[index]

    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = _resize_image(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target
        
        if "oboxes" in target:
            target["oboxes"] = _resize_oboxes(target["oboxes"], (h, w), image.shape[-2:])
        return image, target
    
    def flip_bboxes(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if self._skip_flip or self.training is False or target is None:
            return image, target
            
        direction = self.torch_choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            image = TF.hflip(image)
            
        elif direction == 'vertical':
            image = TF.vflip(image)
             
        elif direction == 'diagonal':
            image = TF.vflip(TF.hflip(image))
        
        if "oboxes" in target:
            target["oboxes"] = _flip_oboxes(target["oboxes"], image.shape[-2:], direction)
        return image, target

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            if "oboxes" in result[i]:
                result[i]["oboxes"] = _resize_oboxes(pred["oboxes"], im_s, o_im_s)
        return result
    
    @staticmethod
    def get_image_transform(train: bool) -> T.Compose:
        return T.Compose([
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
            ),
            T.GaussianBlur(3, sigma=(0.1, 2.0)),
            T.RandomGrayscale(p=0.1),
        ]) if train else None
        
    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        _indent = "\n    "
        format_string += f"{_indent}Normalize(mean={self.image_mean}, std={self.image_std})"
        format_string += f"{_indent}Resize(min_size={self.min_size}, max_size={self.max_size}, mode='bilinear')"
        format_string += "\n)"
        return format_string