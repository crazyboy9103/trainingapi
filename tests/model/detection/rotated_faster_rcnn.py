import pytest
from torch import nn
from trainingapi.model.detection.rotated_faster_rcnn import builder

# Parameters for testing
backbone_types = ["resnet50", "resnet18", "mobilenetv3large", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "unknown_type"]
pretrained_flags = [True, False]
pretrained_backbone_flags = [True, False]
trainable_backbone_layers_options = [0, 3, 5]
freeze_bn_options = [True, False]
returned_layers_options = [
    [1], 
    [1, 2], 
    [1, 2, 3], 
    [1, 2, 3, 4]
]
num_classes = 10
# Define anchor sizes, aspect ratios, and angles to match the number of returned layers
configurations = [
    {
        'returned_layers': [1],
        'anchor_sizes': [(32,), (64,)],
        'aspect_ratios': [(0.5,), (1.0,)],
        'angles': [(0,), (90,)]
    },
    {
        'returned_layers': [1, 2],
        'anchor_sizes': [(32,), (64,), (128,)],
        'aspect_ratios': [(0.5,), (1.0,), (2.0,)],
        'angles': [(0,), (90,), (180,)]
    },
    {
        'returned_layers': [1, 2, 3],
        'anchor_sizes': [(32,), (64,), (128,), (256,)],
        'aspect_ratios': [(0.5,), (1.0,), (2.0,), (3.0,)],
        'angles': [(0,), (90,), (180,), (270,)]
    },
    {
        'returned_layers': [1, 2, 3, 4],
        'anchor_sizes': [(32,), (64,), (128,), (256,), (512,)],
        'aspect_ratios': [(0.5,), (1.0,), (2.0,), (3.0,), (4.0,)],
        'angles': [(0,), (90,), (180,), (270,), (305,)]
    }
]

# Parametrize the test function to cover multiple scenarios
@pytest.mark.parametrize("backbone_type", backbone_types)
@pytest.mark.parametrize("pretrained", pretrained_flags)
@pytest.mark.parametrize("pretrained_backbone", pretrained_backbone_flags)
@pytest.mark.parametrize("trainable_layers", trainable_backbone_layers_options)
@pytest.mark.parametrize("freeze_bn", freeze_bn_options)
@pytest.mark.parametrize("config", configurations)
def test_builder_configurations(backbone_type, pretrained, pretrained_backbone, trainable_layers, freeze_bn, config):
    returned_layers = config['returned_layers']
    anchor_sizes = config['anchor_sizes']
    aspect_ratios = config['aspect_ratios']
    angles = config['angles']
    
    # Exception handling for unknown backbone types
    if backbone_type == "unknown_type":
        with pytest.raises(KeyError):
            builder(pretrained=pretrained, pretrained_backbone=pretrained_backbone, num_classes=num_classes,
                    trainable_backbone_layers=trainable_layers, freeze_bn=freeze_bn, backbone_type=backbone_type,
                    returned_layers=returned_layers, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios, angles=angles)
    else:
        # Attempt to build the model with given parameters
        result = builder(pretrained=pretrained, pretrained_backbone=pretrained_backbone, num_classes=num_classes,
                         trainable_backbone_layers=trainable_layers, freeze_bn=freeze_bn, backbone_type=backbone_type,
                         returned_layers=returned_layers, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios, angles=angles)
        # Check if the result is an instance of nn.Module, indicating successful construction
        assert isinstance(result, nn.Module), f"Builder should return an instance of nn.Module for backbone type {backbone_type}"
