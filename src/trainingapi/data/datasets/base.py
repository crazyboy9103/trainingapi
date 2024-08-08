from typing import Callable, List, Any, Union, Optional, Tuple, Dict

import albumentations as A
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.io import read_image, ImageReadMode

class BaseDataset(Dataset):
    def __init__(
        self,
        image_paths: Optional[List[str]] = None,
        load_image_paths_kwargs: Optional[Dict[str, Any]] = None,
        anns: Optional[List[Any]] = None,
        load_anns_kwargs: Optional[Dict[str, Any]] = None,
        transforms: Optional[Union[A.Compose, T.Compose, Callable]] = None,
        image_mode: ImageReadMode = ImageReadMode.RGB
    ):
        """
        Initialize the dataset with images, annotations, and optional transformations.

        If image_paths and anns are not given, load_image_paths and load_anns are called, 
        assuming that they are implemented in subclasses. 
        If they are not given and the methods are not implemented, it will invoke an error.
         
        Args:
            image_paths (Optional[List[str]]): Absolute paths to images.
            load_image_paths_kwargs (Optional[Dict[str, Any]]): Arguments for loading image paths.
            anns (Optional[List[Any]]): Annotations corresponding to the images.
            load_anns_kwargs (Optional[Dict[str, Any]]): Arguments for loading annotations.
            transforms (Optional[Union[A.Compose, T.Compose, Callable]]): Transforms to apply to each (image, ann) pair.
            image_mode (ImageReadMode): Mode to read the images (e.g., RGB, GRAY, RGBA).
        """
        super().__init__()
        self.image_paths = image_paths or self.load_image_paths(**(load_image_paths_kwargs or {}))
        self.anns = anns or self.load_anns(**(load_anns_kwargs or {}))
        assert len(self.image_paths) == len(self.anns), "image_paths and anns must be of same length"
        self.transforms = transforms
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, index: int) -> Tuple[Tensor, Any]:
        image = read_image(self.image_paths[index], self.image_mode)
        ann = self.anns[index]

        if self.transforms:
            image, ann = self.transforms(image, ann)

        return image, ann

    def load_anns(self, **kwargs) -> List[Any]:
        """Load annotations. Must be implemented by subclasses if anns argument is not given."""
        raise NotImplementedError("load_anns method must be implemented by subclasses.")

    def load_image_paths(self, **kwargs) -> List[str]:
        """Load image paths. Must be implemented by subclasses, if image_paths argument is not given."""
        raise NotImplementedError("load_image_paths method must be implemented by subclasses.")
