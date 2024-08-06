from typing import List, Optional, Tuple, Dict, Callable
from functools import partial

import torch
from torch import nn

from torchvision import models
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision.models.efficientnet import EfficientNet_B1_Weights
from torchvision.models.efficientnet import EfficientNet_B2_Weights
from torchvision.models.efficientnet import EfficientNet_B3_Weights

from trainingapi.model.detection.rpn import RotatedRegionProposalNetwork
from trainingapi.model.detection.rpn import RotatedRPNHead
from trainingapi.model.detection.heads import RotatedFasterRCNNRoIHead
from trainingapi.data.transforms import RotatedFasterRCNNTransform
from trainingapi.model.ops.roi_align_rotated import MultiScaleRotatedRoIAlign
from trainingapi.model.ops.anchors_rotated import RotatedAnchorGenerator

from .generalized_rcnn import GeneralizedRCNN

class RotatedFasterRCNN(GeneralizedRCNN):
    def __init__(
        self, 
        backbone: nn.Module,                
        num_classes: int = 16,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool: nn.Module = None,
        box_head: nn.Module = None,
        box_predictor: nn.Module = None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ) -> None:
        out_channels = backbone.out_channels
        
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = TwoMLPHead(in_channels=out_channels * resolution * resolution, representation_size=1024)

        if box_predictor is None:
            box_predictor = RotatedFasterRCNNPredictor(in_channels=1024, num_classes=num_classes)
            
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = RotatedFasterRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
        
        rpn_head = RotatedRPNHead(in_channels = out_channels, num_anchors = rpn_anchor_generator.num_anchors_per_location()[0], conv_depth = 1, bbox_dim = 6)
        rpn = RotatedRegionProposalNetwork(
            anchor_generator = rpn_anchor_generator, 
            head = rpn_head,
            # Faster-RCNN Training
            fg_iou_thresh = rpn_fg_iou_thresh,
            bg_iou_thresh = rpn_bg_iou_thresh,
            batch_size_per_image = rpn_batch_size_per_image,
            positive_fraction = rpn_positive_fraction,
            # Faster-RCNN Inference
            pre_nms_top_n = dict(
                training=rpn_pre_nms_top_n_train, 
                testing=rpn_pre_nms_top_n_test
            ),
            post_nms_top_n = dict(
                training=rpn_post_nms_top_n_train, 
                testing=rpn_post_nms_top_n_test
            ),
            nms_thresh = rpn_nms_thresh,
            score_thresh = rpn_score_thresh
        )
        
        roi_heads = RotatedFasterRCNNRoIHead(
            # Box
            box_roi_pool = box_roi_pool,
            box_head = box_head,
            box_predictor = box_predictor,
            # Rotated Faster R-CNN training
            fg_iou_thresh = box_fg_iou_thresh,
            bg_iou_thresh = box_bg_iou_thresh,
            batch_size_per_image = box_batch_size_per_image,
            positive_fraction = box_positive_fraction,
            bbox_reg_weights = bbox_reg_weights,
            # Faster R-CNN inference
            score_thresh = box_score_thresh,
            box_nms_thresh = box_nms_thresh,
            detections_per_img = box_detections_per_img,
        )
                
        super(RotatedFasterRCNN, self).__init__(backbone, transform, rpn, roi_heads)

class RotatedFasterRCNNPredictor(nn.Module):
    """
    Standard classification + Oriented bounding box regression layers represented as cx, cy, w, h, cos(a), sin(a)

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.obbox_pred = nn.Linear(in_channels, num_classes * 6)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        obbox_deltas = self.obbox_pred(x)
        return scores, obbox_deltas
    
def builder(
    *,
    pretrained,
    pretrained_backbone,
    num_classes,
    trainable_backbone_layers,
    backbone_type,
    returned_layers,
    freeze_bn = False, 
    anchor_sizes=((8, 16, 32, 64, 128),) * 5, 
    aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    angles=((0, 60, 120, 180, 240, 300),) * 5,
    **kwargs
):
    weights, weights_backbone = get_weights(pretrained, pretrained_backbone, backbone_type)
    
    is_trained = weights is not None or weights_backbone is not None
    backbone_norm_layer = nn.BatchNorm2d if not (freeze_bn or is_trained) else misc_nn_ops.FrozenBatchNorm2d
    trainable_layers = validate_trainable_layers(trainable_backbone_layers, backbone_type, pretrained or pretrained_backbone)

    backbone, feature_map_names = create_backbone(backbone_type, trainable_layers, returned_layers, backbone_norm_layer)
    model = build_model(backbone, feature_map_names, num_classes, anchor_sizes, aspect_ratios, angles, **kwargs)

    if weights is not None or weights_backbone is not None:
        load_model_weights(model, weights, weights_backbone)

    return model

def get_weights(pretrained, pretrained_backbone, backbone_type):
    if pretrained:
        return get_pretrained_weights(backbone_type), None
    if pretrained_backbone:
        return None, get_pretrained_backbone_weights(backbone_type)
    return None, None

def get_pretrained_weights(backbone_type):
    weights_lookup = {
        "resnet50": FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        "mobilenetv3large": FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    }
    return weights_lookup.get(backbone_type, None)

def get_pretrained_backbone_weights(backbone_type):
    backbone_weights_lookup = {
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "mobilenetv3large": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b1": EfficientNet_B1_Weights.IMAGENET1K_V1,
        "efficientnet_b2": EfficientNet_B2_Weights.IMAGENET1K_V1,
        "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
    }
    return backbone_weights_lookup.get(backbone_type, None)

def validate_trainable_layers(trainable_layers, backbone_type, is_trained):
    max_layers = {
        "resnet50": 5, "resnet18": 5, "mobilenetv3large": 6,
        "efficientnet_b0": 5, "efficientnet_b1": 5, "efficientnet_b2": 5, "efficientnet_b3": 5
    }[backbone_type]
    return min(max_layers, trainable_layers if is_trained else max_layers)

def create_backbone(backbone_type, trainable_layers, returned_layers, norm_layer):
    if "resnet" in backbone_type:
        backbone = models.__dict__[backbone_type](norm_layer=norm_layer)
        return _resnet_fpn_extractor(backbone, trainable_layers=trainable_layers, returned_layers=returned_layers, ), ['0', '1', '2', '3'][:len(returned_layers)]
    
    elif backbone_type == "mobilenetv3large":
        backbone = models.mobilenet_v3_large(norm_layer=norm_layer)
        return _mobilenet_extractor(backbone, fpn=True, trainable_layers=trainable_layers, returned_layers=returned_layers), ['0', '1', '2', '3', '4'][:len(returned_layers)]
    
    elif "efficientnet" in backbone_type:
        backbone = models.__dict__[backbone_type](norm_layer=norm_layer)
        return _efficientnet_extractor(backbone,trainable_layers=trainable_layers, returned_layers=returned_layers), ['0', '1', '2', '3'][:len(returned_layers)]
    
    raise ValueError(f"Unsupported backbone type: {backbone_type}")

def build_model(backbone, feature_map_names, num_classes, anchor_sizes, aspect_ratios, angles, **kwargs):
    num_features = len(feature_map_names) + 1
    assert num_features == len(anchor_sizes) == len(aspect_ratios) == len(angles), "Mismatch in number of features and anchor configurations."
    return RotatedFasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=RotatedAnchorGenerator(anchor_sizes, aspect_ratios, angles),
        box_head=FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256] * 4, [1024], norm_layer=nn.BatchNorm2d),
        box_roi_pool=MultiScaleRotatedRoIAlign(featmap_names=feature_map_names, output_size=7, sampling_ratio=2),
        **kwargs
    )

def load_model_weights(model, weights, weights_backbone):
    trained_weights = weights or weights_backbone
    trained_state_dict = trained_weights.get_state_dict()
    model_state_dict = model.state_dict()
    for k, tensor in model_state_dict.items():
        trained_tensor = trained_state_dict.get(k, None)
        if trained_tensor is not None and tensor.shape == trained_tensor.shape:
            model_state_dict[k] = trained_tensor
        else:
            print(f"[WARN] Skipped loading parameter {k} due to incompatible shapes: required shape is {tensor.shape}")
    model.load_state_dict(model_state_dict, strict=False)
    

def _efficientnet_extractor(
    backbone: models.EfficientNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> nn.Module:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0, 2, 3, 4, 6]
    num_stages = len(stage_indices)

    # find the index of the layer from which we won't freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]
    if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
        raise ValueError(f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ")
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}
    
    in_channels_list = []
    for i in returned_layers:
        layer = backbone[stage_indices[i]]
        if isinstance(layer, misc_nn_ops.Conv2dNormActivation):
            in_channels_list.append(layer.out_channels)

        else:
            in_channels_list.append(layer[-1].out_channels)
            
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=nn.BatchNorm2d
    )
    
rotated_faster_rcnn_resnet50_fpn = partial(builder, backbone_type = "resnet50")
rotated_faster_rcnn_resnet18_fpn = partial(builder, backbone_type = "resnet18")
rotated_faster_rcnn_mobilenetv3_large_fpn = partial(builder, backbone_type = "mobilenetv3large")
rotated_faster_rcnn_efficientnetb0_fpn = partial(builder, backbone_type = "efficientnet_b0")
rotated_faster_rcnn_efficientnetb1_fpn = partial(builder, backbone_type = "efficientnet_b1")
rotated_faster_rcnn_efficientnetb2_fpn = partial(builder, backbone_type = "efficientnet_b2")
rotated_faster_rcnn_efficientnetb3_fpn = partial(builder, backbone_type = "efficientnet_b3")

    # pretrained,
    # pretrained_backbone,
    # num_classes,
    # trainable_backbone_layers,
    # backbone_type,
    # returned_layers,
    # freeze_bn = False, 
    # anchor_sizes=((8, 16, 32, 64, 128,)) * 5,
    # aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    # angles=((0, 60, 120, 180, 240, 300),) * 5,