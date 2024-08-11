import os
import glob
from typing import Literal, Optional, Any, Dict, Tuple, Iterable
from typing_extensions import override 

import torch
from torch import Tensor
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities import rank_zero_warn
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont

try:
    import wandb
except ImportError as e:
    print("Requires wandb, pip install wandb")

if os.name == "nt":
    font_dir = os.path.join(os.environ["WINDIR"], "Fonts")
else:
    font_dir = "/usr/share/fonts/"

ttf_files = glob.glob(os.path.join(font_dir, "**/*.ttf"), recursive=True)
FONT = ImageFont.truetype(ttf_files[0] , size=12)
ANCHOR_TYPE = 'lt'

class RotatedDetectionImageLogger(Callback):
    def __init__(
        self, 
        logging_interval: Optional[Literal["step", "epoch"]],
        color_palette: Dict[int, Tuple[int, int, int]],
        class_map: Dict[int, str]
    ) -> None:
        super().__init__()
        self.logging_interval = logging_interval
        if logging_interval == "step":
            rank_zero_warn("step-wise logging is not recommended as it significantly increases the time taken for validation")

        self.color_palette = color_palette
        self.class_map = class_map
        self.logger = None
        self.num_batches_per_epoch = None
        
    @override
    def on_fit_start(self,  trainer: "L.Trainer", pl_module: "L.LightningModule"):
        if not trainer.loggers:
            raise MisconfigurationException(f"{self.__class__.__name__} requires logger.")
        
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                self.logger = logger
        
        if not self.logger:
            raise MisconfigurationException(f"{self.__class__.__name__} must have WandbLogger")

        self.num_batches_per_epoch = trainer.num_val_batches[0] # assume one validloader

    @override
    def on_validation_batch_end(
        self, 
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        images, anns = batch
        if self.logging_interval == "step":
            self.logger.experiment.log({
                "images": [
                    wandb.Image(pil_image)
                    for pil_image in (
                        plot_image(image, pred, ann, self.color_palette, self.class_map, 0.5) for image, pred, ann in zip(images, outputs, anns)
                    )
                ]
            })
        elif batch_idx == self.num_batches_per_epoch - 1:
            self.logger.experiment.log({
                "images": [
                    wandb.Image(pil_image)
                    for pil_image in (
                        plot_image(image, pred, ann, self.color_palette, self.class_map, 0.5) for image, pred, ann in zip(images, outputs, anns)
                    )
                ]
            })


def plot_image(image: Tensor, output: Dict[str, Any], target: Dict[str, Any], 
               color_palette, class_map, o_score_threshold: float = 0.5):
    image = to_pil_image(image.detach().cpu())
    draw = ImageDraw.Draw(image)

    # Draw detected objects
    if 'oboxes' in output:
        draw_detected_objects(draw, output, color_palette, class_map, o_score_threshold)

    # Draw ground truth objects
    draw_ground_truth_objects(draw, target, color_palette, class_map)
    return image

def draw_detected_objects(draw, output, color_palette, class_map, score_threshold):
    dt_oboxes = output['oboxes'].detach().cpu()
    dt_labels = output['labels'].detach().cpu()
    dt_scores = output['scores'].detach().cpu()
    omask = dt_scores > score_threshold

    for dt_obox, dt_label, dt_score in zip(dt_oboxes[omask], dt_labels[omask], dt_scores[omask]):
        color = color_palette.get(dt_label.item(), (255, 255, 255))
        dt_poly = obb2poly(dt_obox).to(int).tolist()
        draw.polygon(dt_poly, outline=color, width=4)
        
        str_obox = obox_to_str(dt_obox)

        text_to_draw = f'DT[{class_map[dt_label.item()]} {dt_score * 100:.2f}% {str_obox}]'
        rectangle = get_xy_bounds_text(draw, dt_poly[:2], text_to_draw)
        draw.rectangle(rectangle, fill="black")
        draw.text(rectangle[:2], text_to_draw,
                  fill=color, font=FONT, anchor=ANCHOR_TYPE)

def draw_ground_truth_objects(draw, target, color_palette, class_map,):
    for gt_obox, gt_label in zip(target['oboxes'].detach(), target['labels'].detach()):
        color = color_palette.get(gt_label.item(), (255, 255, 255))
        gt_opoly = obb2poly(gt_obox).to(int).tolist()
        draw.polygon(gt_opoly, outline=color)

        str_obox = obox_to_str(gt_obox)

        text_to_draw = f'GT[{class_map[gt_label.item()]} {str_obox}]'
        rectangle = get_xy_bounds_text(draw, gt_opoly[:2], text_to_draw,)
        draw.rectangle(rectangle, fill="black")
        draw.text(rectangle[:2], text_to_draw,
                  fill=color, font=FONT, anchor=ANCHOR_TYPE)

def obox_to_str(obox: torch.Tensor) -> str:
    obox = obox.tolist()
    for i in range(4):
        obox[i] = int(obox[i])
    obox[-1] = round(obox[-1], 1)
    return str(obox)
    
def get_xy_bounds_text(draw: ImageDraw.Draw, top_left: Iterable[int], text: str, padding: int = 2) -> Tuple[int, int, int, int]:
    """
    Calculate the bounding box for the text with padding.

    Args:
        draw (ImageDraw.Draw): The drawing context.
        top_left (Iterable[int]): The top-left position to start the text.
        text (str): The text to be drawn.
        padding (int): The padding around the text.

    Returns:
        Tuple[int, int, int, int]: The bounding box for the text (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = draw.textbbox(top_left, text, font=FONT)
    return x1-padding, y1-padding, x2+padding, y2+padding

def obb2poly(obboxes):
    # torch.split not same as np.split
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    theta = torch.deg2rad(theta)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vector1 = torch.cat([w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = torch.cat([-h/2 * Sin, -h/2 * Cos], dim=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    poly = torch.cat([point1, point2, point3, point4], dim=-1)
    return poly