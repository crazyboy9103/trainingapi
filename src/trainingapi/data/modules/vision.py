from typing import Any, Dict

import lightning.pytorch as L

class VisionDataModule(L.LightningDataModule):
    def __init__(self):
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return super().load_state_dict(state_dict)