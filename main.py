import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import LearningRateFinder, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import v2 as T

import wandb

wandb.require("core")

from trainingapi.data.datasets.mvtec import MVTecDataset
from trainingapi.data.modules.vision import VisionDataModule
from trainingapi.model.detection.modules import RotatedFasterRCNN

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
            load_image_paths_kwargs=dict(image_folder="D:/datasets/mvtec/train/images"),
            load_anns_kwargs=dict(ann_folder="D:/datasets/mvtec/train/annfiles"),
            transforms=t,
        ),
        test_kwargs=dict(
            load_image_paths_kwargs=dict(image_folder="D:/datasets/mvtec/test/images"),
            load_anns_kwargs=dict(ann_folder="D:/datasets/mvtec/test/annfiles"),
            transforms=t,
        ),
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    model = RotatedFasterRCNN(
        lr=0.0001,
        min_size = 480,
        max_size= 640,
        rpn_pre_nms_top_n_train = 2000,
        rpn_pre_nms_top_n_test= 2000,
        rpn_post_nms_top_n_train = 2000,
        rpn_post_nms_top_n_test = 2000,
        rpn_nms_thresh = 0.7,
        rpn_fg_iou_thresh = 0.7,
        rpn_bg_iou_thresh = 0.3,
        rpn_batch_size_per_image = 256,
        rpn_positive_fraction = 0.5,
        rpn_score_thresh = 0.0,
        box_score_thresh = 0.05,
        box_nms_thresh = 0.1,
        box_detections_per_img = 200,
        box_fg_iou_thresh = 0.5,
        box_bg_iou_thresh = 0.5,
        box_batch_size_per_image = 512,
        box_positive_fraction = 0.25,
        _skip_image_transform = True,
    )
    logger = WandbLogger(project="ood", name="test", log_model=False, save_dir=".")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=10,
        precision="32",
        benchmark=True,
        deterministic=True,
        callbacks=[
            # LearningRateFinder(0.00001, 0.001),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        num_sanity_val_steps=0,
        log_every_n_steps=1
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
