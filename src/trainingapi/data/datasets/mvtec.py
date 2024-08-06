from typing import Any, List
from glob import glob

from .base import BaseDataset

class MVTecDataset(BaseDataset):
    CLASSES = ('background', 'nut', 'wood_screw', 'lag_wood_screw', 'bolt',
            'black_oxide_screw', 'shiny_screw', 'short_wood_screw', 'long_lag_screw',
            'large_nut', 'nut2', 'nut1', 'machine_screw',
            'short_machine_screw')
    
    PALETTE = [(255, 255, 255), (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0)]
    
    def load_anns(self, ) -> List[Any]:
        return
                
    def load_image_paths(self, image_dir: str) -> List[str]:
        return 
    