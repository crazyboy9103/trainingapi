from typing import Any, Dict

import lightning.pytorch as L
from torch import device
from torch.utils.data import DataLoader

from functools import partial

def default_collate_fn(batch):
    return tuple(zip(*batch))
class VisionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_cls,
        train_kwargs,
        test_kwargs,
        *,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
        drop_last,
        persistent_workers,
        collate_fn = default_collate_fn
    ):
        super().__init__()
        
        self.data_cls = data_cls
        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs

        self.train_loader = partial(
            DataLoader,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn
        )
        self.test_loader = partial(
            DataLoader,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn
        )

    def prepare_data(self) -> None:
        # Do not assign state here as it is called from the main process
        # Only test if the given kwargs and data_cls work
        train = self.data_cls(self.train_kwargs)
        test = self.data_cls(self.test_kwargs)

        image, ann = train[0]
        assert image is not None and ann is not None
        image, ann = test[0]
        assert image is not None and ann is not None
        del train, test

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train = self.data_cls(**self.train_kwargs)
            self.valid = self.data_cls(**self.test_kwargs)

        if stage == "test":
            self.test = self.data_cls(**self.test_kwargs)

    def train_dataloader(self) -> Any:
        return self.train_loader(self.train)

    def val_dataloader(self) -> Any:
        return self.test_loader(self.valid)

    def test_dataloader(self) -> Any:
        return self.test_loader(self.test)

    def predict_dataloader(self) -> Any:
        return self.test_loader(self.test)

    def transfer_batch_to_device(
        self, batch: Any, device: device, dataloader_idx: int
    ) -> Any:
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_exception(self, exception: BaseException) -> None:
        return super().on_exception(exception)

    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return super().load_state_dict(state_dict)
