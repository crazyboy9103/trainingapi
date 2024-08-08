import lightning.pytorch as L

from torch import optim

from trainingapi.evaluation.benchmark_metrics import RotatedMeanAveragePrecision
from trainingapi.model.detection.rotated_faster_rcnn import rotated_faster_rcnn_resnet50_fpn

class RotatedFasterRCNN(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        
        self.model = rotated_faster_rcnn_resnet50_fpn(
            pretrained=True, 
            pretrained_backbone=False,
            num_classes=13,
            trainable_backbone_layers=5, 
            returned_layers=[1,2,3,4],
            freeze_bn=False, 
            anchor_sizes=((8, 16, 32, 64, 128),) * 5,
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
            angles=((0, 60, 120, 180, 240, 300),) * 5,
        )
        self.lr = lr
        
        self.metric = RotatedMeanAveragePrecision(0.5)
    
    def forward(self, images, targets):
        return self.model(images, targets)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self(images, targets)
        
        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train-{k}', v.item(), prog_bar=True, on_epoch=True)
            
        self.log('train-loss', loss.item(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        _, outputs = self(images, targets)
        self.metric.update(outputs, targets)
        self.log_dict(self.metric, on_epoch=True)

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
    