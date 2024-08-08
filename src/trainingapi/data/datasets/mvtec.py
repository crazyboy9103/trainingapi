import os
from typing import Any, List
import glob

from tqdm import tqdm
import torch

from trainingapi.data.datasets.base import BaseDataset

class MVTecDataset(BaseDataset):
    CLASSES = ('background', 'nut', 'wood_screw', 'lag_wood_screw', 'bolt',
            'black_oxide_screw', 'shiny_screw', 'short_wood_screw', 'long_lag_screw',
            'large_nut', 'nut2', 'nut1', 'machine_screw',
            'short_machine_screw')
    
    PALETTE = [(255, 255, 255), (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0)]
    
    def load_anns(self, ann_folder) -> List[Any]:
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        
        ann_files = sorted(glob.glob(ann_folder + '/*.txt'), key=lambda x: x.split("/")[-1].replace(".txt", ""))
        print(ann_files)
            
        anns = []
        image_id = 0
        for ann_file in tqdm(ann_files, desc="Loading annotations"):
            if os.path.getsize(ann_file) == 0:
                continue
            
            gt_difficulty = []
            gt_oboxes = []
            gt_labels = []
            
            with open(ann_file, "r") as f:
                for line in f.readlines():
                    bbox_info = line.split()
                    obb = torch.tensor([float(x) for x in bbox_info[:5]], dtype=torch.float32)
                    cls_name = bbox_info[5]
                    difficulty = int(bbox_info[6])
                    label = cls_map[cls_name]

                    gt_difficulty.append(difficulty)
                    gt_oboxes.append(obb)
                    gt_labels.append(label)

            if gt_oboxes:
                ann = {
                    'difficulty': torch.tensor(gt_difficulty, dtype=torch.int64),
                    'oboxes': torch.stack(gt_oboxes, dim=0),
                    'labels': torch.tensor(gt_labels, dtype=torch.int64),
                    'image_id': torch.tensor([image_id], dtype=torch.int64),
                    'ann_file': ann_file
                }
                anns.append(ann)
                image_id += 1
                
        return anns
                
    def load_image_paths(self, image_folder: str) -> List[str]:
        image_paths = sorted(glob.glob(image_folder + "/*.png"), key=lambda x: x.split("/")[-1].replace(".png", ""))
        print(image_paths)
        return image_paths

if __name__ == "__main__":
    train_data = MVTecDataset(
        load_image_paths_kwargs=dict(
            image_folder = "D:/datasets/mvtec/train/images"
        ),
        load_anns_kwargs=dict(
            ann_folder = "D:/datasets/mvtec/train/annfiles"
        )
    )
    image, label = train_data[0]
    # print(train_data[0])  