from typing import List, Optional, Dict, Tuple 
from collections import OrderedDict

import torch
from torch import nn
from torch import Tensor 

def _check_for_degenerate_boxes(targets):
    for target_idx, target in enumerate(targets):
        oboxes = target["oboxes"]
        degenerate_oboxes = oboxes[:, 2:4] <= 0
        if degenerate_oboxes.any():
            # print the first degenerate box
            bb_idx = torch.where(degenerate_oboxes.any(dim=1))[0][0]
            degen_bb: List[float] = oboxes[bb_idx].tolist()
            torch._assert(
                False,
                "All bounding boxes should have positive height and width."
                f" Found invalid box {degen_bb} for target at index {target_idx}.",
            )
            
class GeneralizedRCNN(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module,                
        transform: nn.Module,
        rpn: nn.Module,
        roi_heads: nn.Module,
    ) -> None:
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.transform = transform
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    oboxes = target["oboxes"]
                    if isinstance(oboxes, torch.Tensor):
                        torch._assert(
                            len(oboxes.shape) == 2 and oboxes.shape[-1] == 5,
                            f"Expected target boxes to be a tensor of shape [N, 5], got {oboxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(oboxes)}.")
                        
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        if targets is not None:
            _check_for_degenerate_boxes(targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
            
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses
        
        if targets:
            return losses, detections
        
        return detections