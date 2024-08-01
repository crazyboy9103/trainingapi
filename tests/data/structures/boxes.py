import pytest
import torch

from trainingapi.data.structures.boxes import BoxMode, Boxes

@pytest.fixture
def BOUNDS():
    return (400, 500)  # H, W

@pytest.fixture
def XYXY_ABS_BOXES():
    return torch.tensor([
        [0.0, 50.0, 100.0, 200.0], 
        [10.0, 20.0, 30.0, 40.0]
    ])

@pytest.fixture
def XYXY_REL_BOXES():
    return torch.tensor([
        [0.0, 0.125, 0.2, 0.5],  # [0/500, 50/400, 100/500, 200/400]
        [0.02, 0.05, 0.06, 0.1]  # [10/500, 20/400, 30/500, 40/400]
    ])

@pytest.fixture
def XYWH_ABS_BOXES():
    return torch.tensor([
        [0.0, 50.0, 100.0, 150.0],  # Converted to (x, y, width, height)
        [10.0, 20.0, 20.0, 20.0]
    ])

@pytest.fixture
def XYWH_REL_BOXES():
    return torch.tensor([
        [0.0, 0.125, 0.2, 0.375],  # [0/500, 50/400, 100/500, 150/400]
        [0.02, 0.05, 0.04, 0.05]   # [10/500, 20/400, 20/500, 20/400]
    ])

@pytest.fixture
def CXCYWH_ABS_BOXES():
    return torch.tensor([
        [50.0, 125.0, 100.0, 150.0],  # Center (x, y), width, height
        [20.0, 30.0, 20.0, 20.0]
    ])

@pytest.fixture
def CXCYWH_REL_BOXES():
    return torch.tensor([
        [0.1, 0.3125, 0.2, 0.375],  # [50/500, 125/400, 100/500, 150/400]
        [0.04, 0.075, 0.04, 0.05]   # [20/500, 30/400, 20/500, 20/400]
    ])

@pytest.fixture
def XYXY_ABS_BOXES_1D():
    return torch.tensor([0.0, 50.0, 100.0, 200.0])

@pytest.fixture
def XYXY_REL_BOXES_1D():
    return torch.tensor([0.0, 0.125, 0.2, 0.5])

@pytest.fixture
def XYWH_ABS_BOXES_1D():
    return torch.tensor([0.0, 50.0, 100.0, 150.0])

@pytest.fixture
def XYWH_REL_BOXES_1D():
    return torch.tensor([0.0, 0.125, 0.2, 0.375])

@pytest.fixture
def CXCYWH_ABS_BOXES_1D():
    return torch.tensor([50.0, 125.0, 100.0, 150.0])

@pytest.fixture
def CXCYWH_REL_BOXES_1D():
    return torch.tensor([0.1, 0.3125, 0.2, 0.375])

def test_len(XYXY_ABS_BOXES):
    boxes = Boxes(XYXY_ABS_BOXES, BoxMode.XYXY_ABS)
    assert len(boxes) == XYXY_ABS_BOXES.size(0)

def test_device(XYXY_ABS_BOXES):
    boxes = Boxes(XYXY_ABS_BOXES, BoxMode.XYXY_ABS)
    assert boxes.device() == XYXY_ABS_BOXES.device

def test_clone(XYXY_ABS_BOXES):
    boxes = Boxes(XYXY_ABS_BOXES, BoxMode.XYXY_ABS)
    cloned_boxes = boxes.clone()
    assert cloned_boxes.boxes.equal(XYXY_ABS_BOXES)

def test_to_cpu(XYXY_ABS_BOXES):
    boxes = Boxes(XYXY_ABS_BOXES, BoxMode.XYXY_ABS)
    to_device = torch.device("cpu")
    to_boxes = boxes.to(to_device)
    assert to_boxes.boxes.device == to_device

def test_to_cuda(XYXY_ABS_BOXES):
    boxes = Boxes(XYXY_ABS_BOXES, BoxMode.XYXY_ABS)
    if torch.cuda.is_available():
        to_device = torch.device("cuda:0")
        to_boxes = boxes.to(to_device)
        assert to_boxes.boxes.device == to_device

def test_area(XYXY_ABS_BOXES, CXCYWH_ABS_BOXES):
    boxes_xyxy = Boxes(XYXY_ABS_BOXES, BoxMode.XYXY_ABS)
    boxes_cxcywh = Boxes(CXCYWH_ABS_BOXES, BoxMode.CXCYWH_ABS)
    assert boxes_xyxy.area().equal(torch.prod(XYXY_ABS_BOXES[:, 2:] - XYXY_ABS_BOXES[:, :2], dim=1))
    assert boxes_cxcywh.area().equal(torch.prod(CXCYWH_ABS_BOXES[:, 2:], dim=1))

def test_convert_mode(XYXY_ABS_BOXES, XYXY_REL_BOXES, XYWH_ABS_BOXES, XYWH_REL_BOXES, CXCYWH_ABS_BOXES, CXCYWH_REL_BOXES, BOUNDS):
    xyxy_abs_boxes = Boxes(XYXY_ABS_BOXES, BoxMode.XYXY_ABS, BOUNDS)
    xyxy_rel_boxes = Boxes(XYXY_REL_BOXES, BoxMode.XYXY_REL, BOUNDS)
    xywh_abs_boxes = Boxes(XYWH_ABS_BOXES, BoxMode.XYWH_ABS, BOUNDS)
    xywh_rel_boxes = Boxes(XYWH_REL_BOXES, BoxMode.XYWH_REL, BOUNDS)
    cxcywh_abs_boxes = Boxes(CXCYWH_ABS_BOXES, BoxMode.CXCYWH_ABS, BOUNDS)
    cxcywh_rel_boxes = Boxes(CXCYWH_REL_BOXES, BoxMode.CXCYWH_REL, BOUNDS)
    
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYXY_ABS) == xyxy_abs_boxes
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYXY_REL).boxes.allclose(xyxy_rel_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYWH_REL).boxes.allclose(xywh_rel_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYWH_ABS).boxes.allclose(xywh_abs_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.CXCYWH_ABS).boxes.allclose(cxcywh_abs_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.CXCYWH_REL).boxes.allclose(cxcywh_rel_boxes.boxes, atol=1e-5)

    assert cxcywh_abs_boxes.convert_mode(BoxMode.CXCYWH_ABS) == cxcywh_abs_boxes
    assert cxcywh_abs_boxes.convert_mode(BoxMode.CXCYWH_REL).boxes.allclose(cxcywh_rel_boxes.boxes, atol=1e-5)
    assert cxcywh_abs_boxes.convert_mode(BoxMode.XYWH_ABS).boxes.allclose(xywh_abs_boxes.boxes, atol=1e-5)
    assert cxcywh_abs_boxes.convert_mode(BoxMode.XYXY_ABS).boxes.allclose(xyxy_abs_boxes.boxes, atol=1e-5)

def test_area_1D(XYXY_ABS_BOXES_1D, CXCYWH_ABS_BOXES_1D):
    boxes_xyxy = Boxes(XYXY_ABS_BOXES_1D, BoxMode.XYXY_ABS)
    boxes_cxcywh = Boxes(CXCYWH_ABS_BOXES_1D, BoxMode.CXCYWH_ABS)
    assert boxes_xyxy.area().equal(torch.prod(XYXY_ABS_BOXES_1D[2:] - XYXY_ABS_BOXES_1D[:2])[None])
    assert boxes_cxcywh.area().equal(torch.prod(CXCYWH_ABS_BOXES_1D[2:])[None])

def test_convert_mode_1D(XYXY_ABS_BOXES_1D, XYXY_REL_BOXES_1D, XYWH_ABS_BOXES_1D, XYWH_REL_BOXES_1D, CXCYWH_ABS_BOXES_1D, CXCYWH_REL_BOXES_1D, BOUNDS):
    xyxy_abs_boxes = Boxes(XYXY_ABS_BOXES_1D, BoxMode.XYXY_ABS, BOUNDS)
    xyxy_rel_boxes = Boxes(XYXY_REL_BOXES_1D, BoxMode.XYXY_REL, BOUNDS)
    xywh_abs_boxes = Boxes(XYWH_ABS_BOXES_1D, BoxMode.XYWH_ABS, BOUNDS)
    xywh_rel_boxes = Boxes(XYWH_REL_BOXES_1D, BoxMode.XYWH_REL, BOUNDS)
    cxcywh_abs_boxes = Boxes(CXCYWH_ABS_BOXES_1D, BoxMode.CXCYWH_ABS, BOUNDS)
    cxcywh_rel_boxes = Boxes(CXCYWH_REL_BOXES_1D, BoxMode.CXCYWH_REL, BOUNDS)

    # Testing conversion from XYXY_ABS
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYXY_ABS) == xyxy_abs_boxes
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYXY_REL).boxes.allclose(xyxy_rel_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYWH_REL).boxes.allclose(xywh_rel_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.XYWH_ABS).boxes.allclose(xywh_abs_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.CXCYWH_ABS).boxes.allclose(cxcywh_abs_boxes.boxes, atol=1e-5)
    assert xyxy_abs_boxes.convert_mode(BoxMode.CXCYWH_REL).boxes.allclose(cxcywh_rel_boxes.boxes, atol=1e-5)

    # Testing conversion from CXCYWH_ABS
    assert cxcywh_abs_boxes.convert_mode(BoxMode.CXCYWH_ABS) == cxcywh_abs_boxes
    assert cxcywh_abs_boxes.convert_mode(BoxMode.CXCYWH_REL).boxes.allclose(cxcywh_rel_boxes.boxes, atol=1e-5)
    assert cxcywh_abs_boxes.convert_mode(BoxMode.XYWH_ABS).boxes.allclose(xywh_abs_boxes.boxes, atol=1e-5)
    assert cxcywh_abs_boxes.convert_mode(BoxMode.XYXY_ABS).boxes.allclose(xyxy_abs_boxes.boxes, atol=1e-5)

    # Testing conversion from XYWH_ABS
    assert xywh_abs_boxes.convert_mode(BoxMode.XYWH_ABS) == xywh_abs_boxes
    assert xywh_abs_boxes.convert_mode(BoxMode.XYWH_REL).boxes.allclose(xywh_rel_boxes.boxes, atol=1e-5)
    assert xywh_abs_boxes.convert_mode(BoxMode.XYXY_ABS).boxes.allclose(xyxy_abs_boxes.boxes, atol=1e-5)
    assert xywh_abs_boxes.convert_mode(BoxMode.CXCYWH_ABS).boxes.allclose(cxcywh_abs_boxes.boxes, atol=1e-5)

    # Testing conversion from XYWH_REL
    assert xywh_rel_boxes.convert_mode(BoxMode.XYWH_REL) == xywh_rel_boxes
    assert xywh_rel_boxes.convert_mode(BoxMode.XYWH_ABS).boxes.allclose(xywh_abs_boxes.boxes, atol=1e-5)
    assert xywh_rel_boxes.convert_mode(BoxMode.XYXY_ABS).boxes.allclose(xyxy_abs_boxes.boxes, atol=1e-5)
    assert xywh_rel_boxes.convert_mode(BoxMode.CXCYWH_ABS).boxes.allclose(cxcywh_abs_boxes.boxes, atol=1e-5)