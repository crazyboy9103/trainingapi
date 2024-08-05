import pytest
from torch import nn

from trainingapi.model.detection.rotated_faster_rcnn import builder, get_weights, create_backbone, build_model, load_model_weights

# Mocks for weights to be used in tests
class MockWeights:
    @staticmethod
    def get_state_dict():
        return {'layer1.weight': 'weights'}

# Test builder function for different backbone configurations
@pytest.mark.parametrize("backbone_type,expected_exception", [
    ("resnet50", None),
    ("resnet18", None),
    ("mobilenetv3large", None),
    ("efficientnet_b0", None),
    ("efficientnet_b1", None),
    ("efficientnet_b2", None),
    ("efficientnet_b3", None),
    ("unknown_type", ValueError)
])
def test_builder_backbone_type(mocker, backbone_type, expected_exception):
    # Mock the internal methods called within the builder
    mocker.patch('your_module_name.get_weights', return_value=(None, None))
    mocker.patch('your_module_name.create_backbone', return_value=(nn.Module(), ['0', '1']))
    mocker.patch('your_module_name.build_model', return_value=nn.Module())

    # Set parameters
    params = {
        "pretrained": True,
        "pretrained_backbone": False,
        "num_classes": 10,
        "trainable_backbone_layers": 3,
        "freeze_bn": True,
        "backbone_type": backbone_type,
        "returned_layers": [1, 2, 3]
    }

    # Test for expected exceptions
    if expected_exception:
        with pytest.raises(expected_exception):
            builder(**params)
    else:
        assert isinstance(builder(**params), nn.Module), "Builder should return a nn.Module object"

# Test weight loading
def test_get_weights_pretrained(mocker):
    mocker.patch('your_module_name.get_pretrained_weights', return_value=MockWeights())
    mocker.patch('your_module_name.get_pretrained_backbone_weights', return_value=None)
    weights, backbone_weights = get_weights(True, False, "resnet50")
    assert weights is not None, "Weights should not be None when pretrained is True"
    assert backbone_weights is None, "Backbone weights should be None when pretrained_backbone is False"

# Test model building function
def test_build_model(mocker):
    backbone = nn.Module()
    mocker.patch.object(backbone, 'out_channels', 256, create=True)
    model = build_model(backbone, 10, ['0', '1'], ((32, 64),), ((1.0,),), ((0, 90),), nn.BatchNorm2d)
    assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"

# Example test for loading weights
def test_load_model_weights():
    model = nn.Module()
    model.load_state_dict = lambda x, strict: None  # mock load_state_dict method
    load_model_weights(model, MockWeights())
    # Further assertions can be made depending on the modifications to the model's state dict
