from torch.utils.data import Dataset
from PIL import Image
from os import path as ph
from torch import Tensor
import pandas as pd
import numpy as np


class SegmentationDataSet(Dataset):
    def __init__(
        self,
        csv_file: str,
        image_column: str,
        image_sufix: str,
        mask_sufix: str,
        image_dir: str,
        mask_dir: str,
        image_size: int,
    ) -> None:
        self.images_id = pd.read_csv(csv_file)[image_column].to_list()
        self.image_sufix = image_sufix
        self.mask_sufix = mask_sufix
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.images_id)

    def __getitem__(self, index: int) -> tuple:
        image_id = self.images_id[index]
        image_shape = (self.image_size, self.image_size)
        image_path = f"{image_id}{self.image_sufix}"
        mask_path = f"{image_id}{self.mask_sufix}"
        image = (
            np.array(
                Image.open(image_path).convert("RGB").resize(size=image_shape)
            ).astype("float32")
            / 255
        )
        mask = (
            np.array(
                Image.open(mask_path).convert("L").resize(size=image_shape)
            ).astype("float32")
            / 255
        )

        return image, mask
