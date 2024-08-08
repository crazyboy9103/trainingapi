from typing import List, Union, Dict

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.prints import rank_zero_warn

from trainingapi.model.ops.box_iou_rotated import box_iou_rotated

class RotatedMeanAveragePrecision(Metric):
    def __init__(self, iou_threshold: Union[float, List[float]] = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold if isinstance(iou_threshold, list) else [iou_threshold]
        
        self.add_state("target_boxes", default=[], dist_reduce_fx=None)
        self.add_state("target_labels", default=[], dist_reduce_fx=None)
        
        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("pred_scores", default=[], dist_reduce_fx=None)
        
    def update(
        self, 
        preds: List[Dict[str, Tensor]], 
        targets: List[Dict[str, Tensor]]
    ) -> None:
        """
        Args: 
            preds: 
                List of predictions for each image. Each prediction must contain the following keys:
                    - "oboxes": Predicted bounding boxes in format (cx, cy, w, h, theta in `degrees`)
                    - "labels": Predicted labels
                    - "scores": Predicted scores
                    
            targets: 
                List of gt targets for each image. Each target must contain the following keys:
                    - "oboxes": gt bounding boxes in the same format as pred
                    - "labels": gt labels
        """
        assert len(preds) == len(targets)
        for pred, target in zip(preds, targets):
            self.pred_boxes.append(pred["oboxes"].detach().cpu())
            self.pred_labels.append(pred["labels"].detach().cpu())
            self.pred_scores.append(pred["scores"].detach().cpu())
            
            self.target_boxes.append(target["oboxes"].detach().cpu())
            self.target_labels.append(target["labels"].detach().cpu())
    
    def compute(self):
        classes = torch.unique(torch.cat(self.pred_labels)).numpy()
        
        average_metrics = torch.zeros(6)
        metrics_by_iou_threshold = {}
        metrics_by_class = {}
        for iou_threshold in self.iou_threshold:
            metrics = torch.zeros(6)
            metrics_by_class[iou_threshold] = {}
            
            for class_id in classes:
                class_metrics = torch.tensor(self._compute_class_metrics(class_id, iou_threshold))
                metrics += class_metrics / len(classes)
                metrics_by_class[iou_threshold][class_id] = {
                    "Precision": class_metrics[0].item(),
                    "Recall": class_metrics[1].item(),
                    "F1-Score": class_metrics[2].item(),
                    "AP": class_metrics[3].item(),
                    "mIoU": class_metrics[4].item(),
                    "SoAP": class_metrics[5].item()
                }
                
            metrics_by_iou_threshold[iou_threshold] = {
                "Precision": metrics[0].item(),
                "Recall": metrics[1].item(),
                "F1-Score": metrics[2].item(),
                "mAP": metrics[3].item(),
                "mIoU": metrics[4].item(),
                "SoAP": metrics[5].item(),
            }        
            
            metrics[3] *= iou_threshold

            average_metrics += metrics
        
        average_metrics /= len(self.iou_threshold)
        
        # mAP = sum(iou threshold * AP@iou / sum(iou_thresholds))
        average_metrics[3] *= len(self.iou_threshold)
        average_metrics[3] /= sum(self.iou_threshold)
        
        average_metrics = {
            "Precision": average_metrics[0].item(),
            "Recall": average_metrics[1].item(),
            "F1-Score": average_metrics[2].item(),
            "mAP": average_metrics[3].item(),
            "mIoU": average_metrics[4].item(),
            "SoAP": average_metrics[5].item(),
        }
        return average_metrics, metrics_by_iou_threshold, metrics_by_class
        
    def _compute_class_metrics(self, class_id, iou_threshold):
        gt_mask = [label == class_id for label in self.target_labels]
        pred_mask = [label == class_id for label in self.pred_labels]
        
        class_gt_bboxes = [boxes[mask] for boxes, mask in zip(self.target_boxes, gt_mask)]
        class_dt_bboxes = [boxes[mask] for boxes, mask in zip(self.pred_boxes, pred_mask)]
        class_dt_scores = [scores[mask] for scores, mask in zip(self.pred_scores, pred_mask)]
        
        TP = []
        FP = []
        dt_ious = []
        gt_angles = []
        dt_angles = []
        dt_scores_for_sort = []
        # This will be used as TP + FN
        total_num_gt_bboxes = 0
        
        for image_idx, (gt_bboxes, dt_bboxes, dt_scores) in enumerate(zip(class_gt_bboxes, class_dt_bboxes, class_dt_scores)):
            num_gt_bboxes = gt_bboxes.shape[0]
            num_dt_bboxes = dt_bboxes.shape[0]

            if num_gt_bboxes == 0 or num_dt_bboxes == 0:
                continue
            
            total_num_gt_bboxes += num_gt_bboxes
            # Sort detections by score in descending order
            sorted_indices = torch.argsort(dt_scores, descending=True)
            
            dt_bboxes = dt_bboxes[sorted_indices]
            dt_scores = dt_scores[sorted_indices]
            
            ious = box_iou_rotated(dt_bboxes, gt_bboxes, angle_aware=False)
            max_ious = ious.max(dim=1)
            # To keep track of which ground truth boxes have been assigned
            # GT boxes can only be assigned once
            assigned_gt_boxes = set()
            for i, (dt_bbox, dt_score) in enumerate(zip(dt_bboxes, dt_scores)):
                # max_iou = -1
                # best_gt_idx = None
                # for gt_idx, gt_bbox in enumerate(gt_bboxes):    
                #     iou = box_iou_rotated(dt_bbox[None, :], gt_bbox[None, :], angle_aware=False).item()
                #     if iou > max_iou:
                #         max_iou = iou
                #         best_gt_idx = gt_idx
                max_iou = max_ious.values[i].item()
                best_gt_idx = max_ious.indices[i].item() # .item() is crucial as it is added to set
                
                dt_scores_for_sort.append(dt_score)
                if max_iou >= iou_threshold:
                    if best_gt_idx in assigned_gt_boxes:
                        FP.append(1)
                        TP.append(0)
                        
                    else:
                        assigned_gt_boxes.add(best_gt_idx)
                        
                        FP.append(0)
                        TP.append(1)
                        dt_ious.append(max_iou)
                        
                        # Compute metric for angle 
                        gt_angles.append(gt_bboxes[best_gt_idx][4])
                        dt_angles.append(dt_bbox[4])
                        
                else:
                    FP.append(1)
                    TP.append(0)
        
        # If there are no true positives, return 0 for all metrics
        if not sum(TP):
            return 0, 0, 0, 0, 0, 0
        
        sorted_indices = torch.argsort(torch.as_tensor(dt_scores_for_sort), descending=True)
        TP = torch.as_tensor(TP)[sorted_indices]
        FP = torch.as_tensor(FP)[sorted_indices]
        dt_ious = torch.as_tensor(dt_ious)    
        gt_angles = torch.as_tensor(gt_angles)
        dt_angles = torch.as_tensor(dt_angles)
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_num_gt_bboxes + 1e-10)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-10)
        
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        f1_max_idx = torch.argmax(f1_scores)
        recall = recalls[f1_max_idx].item()
        precision = precisions[f1_max_idx].item()
        f1_score = f1_scores[f1_max_idx].item()  
        
        # Add 0 and 1 to the start and end of the arrays to make sure the curve starts at (0, 1) and ends at (1, 0)
        # This is required for computing the area under the curve      
        recalls = torch.cat([torch.tensor([0]), recalls, torch.tensor([1])])
        precisions = torch.cat([torch.tensor([1]), precisions, torch.tensor([0])])
        
        average_precision = torch.trapz(precisions, recalls).item()
        
        mean_iou = dt_ious.mean().item()
        
        abs_diff_angle = torch.abs(gt_angles - dt_angles)
        # Score of angular precision (SoAP) 
        # We rather use the mean 
        soap = torch.min(abs_diff_angle, 360 - abs_diff_angle).mean().item()
        return precision, recall, f1_score, average_precision, mean_iou, soap