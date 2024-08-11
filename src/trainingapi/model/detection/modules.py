from typing import Any
import lightning.pytorch as L

from torch import optim
from torch import distributed as dist 

from trainingapi.evaluation.benchmark_metrics import RotatedMeanAveragePrecision
from trainingapi.model.detection.rotated_faster_rcnn import rotated_faster_rcnn_resnet50_fpn

class RotatedFasterRCNN(L.LightningModule):
    def __init__(self, lr, **model_kwargs):
        super().__init__()
        
        self.model = rotated_faster_rcnn_resnet50_fpn(
            num_classes=14,
            anchor_sizes=((8, 16, 32, 64, 128),) * 5,
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
            angles=((0, 60, 120, 180, 240, 300),) * 5,
            **model_kwargs
        )
        self.lr = lr
        
        self.metric = RotatedMeanAveragePrecision(0.5, compute_on_cpu=False)
    
    def forward(self, images, targets):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self(images, targets)
        
        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train-{k}', v.item(), prog_bar=True, on_epoch=True, sync_dist=dist.is_initialized())
            
        self.log('train-loss', loss.item(), prog_bar=True, on_epoch=True, sync_dist=dist.is_initialized())
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        _, outputs = self(images, targets)
        self.metric.update(outputs, targets)
        return outputs # to use it in callback on_validation_epoch_end
    
    def on_validation_epoch_end(self):
        average_metrics, metrics_by_iou_threshold, metrics_by_class = self.metric.compute()
        self.log_dict(average_metrics, sync_dist=dist.is_initialized())
        
        for iou_threshold, metrics in metrics_by_iou_threshold.items():
            self.log_dict({f"{k}@{iou_threshold}": v for k, v in metrics.items()}, sync_dist=dist.is_initialized())
        
        self.metric.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # # following milestones, warmup_iters are arbitrarily chosen
        # first, second = steps_per_epoch * int(self.trainer.max_epochs * 4/6), steps_per_epoch * int(self.trainer.max_epochs * 5/6)
        # warmup_iters = steps_per_epoch * int(self.trainer.max_epochs * 1/6)
        # scheduler = LinearWarmUpMultiStepDecay(optimizer, milestones=[first, second], gamma=1/3, warmup_iters=warmup_iters)
        # scheduler_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        # }
        # return [optimizer], [scheduler_config]
        return optimizer
    