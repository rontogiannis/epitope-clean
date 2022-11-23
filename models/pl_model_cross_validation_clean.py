import os
import sys
import copy

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import esm

from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from torchmetrics import AUROC

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from typing import Callable, Union

d3to1 = {
    "CYS": "C", "ASP": "D", "SER": "S",
    "GLN": "Q", "LYS": "K", "ILE": "I",
    "PRO": "P", "THR": "T", "PHE": "F",
    "ASN": "N", "GLY": "G", "HIS": "H",
    "LEU": "L", "ARG": "R", "TRP": "W",
    "ALA": "A", "VAL": "V", "GLU": "E",
    "TYR": "Y", "MET": "M",
}

class BaseKFoldDataModule(LightningDataModule) :
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None :
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None :
        pass

