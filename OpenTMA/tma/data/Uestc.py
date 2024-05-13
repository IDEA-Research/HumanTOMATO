from .base import BASEDataModule
from .a2m import UESTC
import os
import rich.progress
import pickle as pkl


class UestcDataModule(BASEDataModule):

    def __init__(
        self,
        cfg,
        batch_size,
        num_workers,
        collate_fn=None,
        method_name="vibe",
        phase="train",
        **kwargs
    ):
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
        )
        self.save_hyperparameters(logger=False)
        self.name = "Uestc"

        self.Dataset = UESTC
        self.cfg = cfg

        # self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = 150
        self.njoints = 25
        self.nclasses = 40
        # self.transforms = self._sample_set.transforms
