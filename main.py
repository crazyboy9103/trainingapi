import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, LearningRateFinder

import torch 
from torchvision.transforms import v2 as T
import wandb
wandb.require("core")

from trainingapi.data.modules.vision import VisionDataModule
from trainingapi.data.datasets.mvtec import MVTecDataset
from trainingapi.model.detection.modules import RotatedFasterRCNN

def main():
    t = T.Compose([
        T.ToDtype(torch.float32, scale=True),
    ])
    dm = VisionDataModule(
        MVTecDataset, 
        train_kwargs=dict(
            load_image_paths_kwargs=dict(
                image_folder = "D:/datasets/mvtec/train/images"
            ),
            load_anns_kwargs=dict(
                ann_folder = "D:/datasets/mvtec/train/annfiles"
            ),
            transforms = t
        ),
        test_kwargs=dict(
            load_image_paths_kwargs=dict(
                image_folder = "D:/datasets/mvtec/test/images"
            ),
            load_anns_kwargs=dict(
                ann_folder = "D:/datasets/mvtec/test/annfiles"
            ),
            transforms = t
        ),
        batch_size = 8,
        shuffle = True,
        num_workers = 2,
        pin_memory = True,
        drop_last = False,
        persistent_workers = False, 
    )
    
    model = RotatedFasterRCNN(lr=0.001)
    logger = WandbLogger(project="ood", name="test", log_model=False, save_dir=".")
    
    trainer = L.Trainer(
        accelerator = "gpu", 
        devices = 1,
        logger = logger,
        max_epochs = 10,
        precision="16-mixed",
        benchmark=True,
        deterministic=True,
        callbacks=[
            LearningRateFinder(0.00001, 0.001),
            LearningRateMonitor(logging_interval="epoch")
        ],
        num_sanity_val_steps=0
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()