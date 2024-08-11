import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import LearningRateFinder, LearningRateMonitor
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torchvision.transforms import v2 as T

import wandb

wandb.require("core")

from trainingapi.data.datasets.mvtec import MVTecDataset
from trainingapi.data.modules.vision import VisionDataModule
from trainingapi.model.detection.modules import RotatedFasterRCNN
from trainingapi.callback.experiments import RotatedDetectionImageLogger

def main():
    L.seed_everything(2024)
    
    t = T.Compose(
        [
            T.ToDtype(torch.float32, scale=True),
        ]
    )
    dm = VisionDataModule(
        data_cls=MVTecDataset,
        train_kwargs=dict(
            load_image_paths_kwargs=dict(image_folder="/home/work/.trainingapi/mvtec/train/images"),
            load_anns_kwargs=dict(ann_folder="/home/work/.trainingapi/mvtec/train/annfiles"),
            transforms=t,
        ),
        test_kwargs=dict(
            load_image_paths_kwargs=dict(image_folder="/home/work/.trainingapi/mvtec/test/images"),
            load_anns_kwargs=dict(ann_folder="/home/work/.trainingapi/mvtec/test/annfiles"),
            transforms=t,
        ),
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    model = RotatedFasterRCNN(
        pretrained=True, 
        pretrained_backbone=True,
        trainable_backbone_layers=5, 
        returned_layers=[1,2,3,4],
        freeze_bn=False, 
        lr=0.0001,
        min_size = 480,
        max_size= 640,
        rpn_pre_nms_top_n_train = 2000,
        rpn_pre_nms_top_n_test= 1000,
        rpn_post_nms_top_n_train = 2000,
        rpn_post_nms_top_n_test = 1000,
        rpn_nms_thresh = 0.7,
        rpn_fg_iou_thresh = 0.7,
        rpn_bg_iou_thresh = 0.3,
        rpn_batch_size_per_image = 256,
        rpn_positive_fraction = 0.5,
        rpn_score_thresh = 0.0,
        box_score_thresh = 0.05,
        box_nms_thresh = 0.5,
        box_detections_per_img = 200,
        box_fg_iou_thresh = 0.5,
        box_bg_iou_thresh = 0.5,
        box_batch_size_per_image = 512,
        box_positive_fraction = 0.25,
        _skip_image_transform = True,
    )
    logger = WandbLogger(project="ood", name="test", log_model=False, save_dir=".")

    # model = torch.compile(model)
    devices = find_usable_cuda_devices(-1)
    trainer = L.Trainer(
        # strategy=DDPStrategy(process_group_backend="gloo"),
        accelerator="gpu",
        devices=devices,
        logger=logger,
        max_epochs=36,
        precision="32",
        benchmark=True,
        deterministic=True,
        callbacks=[
            # LearningRateFinder(0.00001, 0.01),
            LearningRateMonitor(logging_interval="epoch"),
            RotatedDetectionImageLogger(logging_interval="epoch", color_palette={
                0: (255, 255, 255), 
                1: (165, 42, 42),
                2: (189, 183, 107),
                3: (0, 255, 0), 
                4: (255, 0, 0),
                5:(138, 43, 226), 
                6: (255, 128, 0), 
                7: (255, 0, 255), 
                8: (0, 255, 255),
                9: (255, 193, 193), 
                10: (0, 51, 153), 
                11: (255, 250, 205), 
                12: (0, 139, 139),
                13: (255, 255, 0)
            },
            class_map = {
                0: 'background', 
                1: 'nut', 
                2: 'wood_screw',
                3: 'lag_wood_screw', 
                4: 'bolt',
                5: 'black_oxide_screw', 
                6: 'shiny_screw', 
                7: 'short_wood_screw', 
                8: 'long_lag_screw',
                9: 'large_nut',
                10: 'nut2', 
                11: 'nut1', 
                12: 'machine_screw',
                13: 'short_machine_screw'
            }
            )
        ],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        # fast_dev_run=True,
        sync_batchnorm=len(devices) > 1
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
