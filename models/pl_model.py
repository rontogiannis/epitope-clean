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

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger

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

# mayhaps remove eventually, added for reproducability
torch.manual_seed(13)

# Multi-Layer Perceptron
class MLP(nn.Module) :
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: int,
    ) :
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
        )

    def forward(self, embeddings) :
        return self.linear(embeddings)

# Language Model
class LM(nn.Module) :
    def __init__(
        self,
        model_name: str,
        layer_count: int,
        embedding_dim: int,
        finetune: bool = False,
    ) :
        super().__init__()

        try :
            esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
        except Exception :
            bar = getattr(esm.pretrained, model_name)
            esm_model, alphabet = bar()

        self.esm_model = esm_model
        self.alphabet = alphabet
        self.layer_count = layer_count
        self.embedding_dim = embedding_dim
        self.finetune = finetune

        # Freeze (or not) the model
        for param in self.esm_model.parameters() :
            param.requires_grad = finetune

    def forward(self, X) :
        if not self.finetune :
            with torch.no_grad() :
                dct = self.esm_model(X, repr_layers=[self.layer_count], return_contacts=True)
        else :
            dct = self.esm_model(X, repr_layers=[self.layer_count], return_contacts=True)

        return dct["representations"][self.layer_count]

class EpitopeModel(nn.Module) :
    def __init__(
        self,
        lm: LM,
        hidden_dim: int,
        dropout: float,
    ) :
        super().__init__()

        self.lm = lm if not lm.finetune else copy.deepcopy(lm)
        self.mlp = MLP(
            input_dim = lm.embedding_dim,
            hidden_dim = hidden_dim,
            dropout = dropout,
        )

    def forward(self, X) :
        out = self.lm(X)
        out = self.mlp(out)

        return out

# Simple epitope model, ESM2 + MLP
class LitModule(pl.LightningModule) :
    def __init__(
        self,
        criterion: Callable = nn.BCEWithLogitsLoss,
        lr: float = 0.001,
        model: Union[EpitopeModel, None] = None,
    ) :
        super().__init__()

        self.criterion = criterion
        self.lr = lr

        self.model = model

        self.train_auc = AUROC()
        self.valid_auc = AUROC()
        self.test_auc = AUROC()

        self.test_preds = []
        self.test_reals = []

    def configure_optimizers(self) :
        return torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def _shared_step(self, batch, batch_idx) :
        X, mask, y = batch
        out = self.model(X)
        mask = mask.flatten()
        out = out.flatten()
        out = torch.masked_select(out, mask)
        y = y.flatten()
        y = torch.masked_select(y, mask)
        loss = self.criterion(out, y.float())

        return loss, out, y

    def on_train_epoch_start(self) :
        self.train_auc = AUROC()

    def on_validation_start(self) :
        self.valid_auc = AUROC()

    def on_test_start(self) :
        self.test_auc = AUROC()
        self.test_preds = []
        self.test_reals = []

    def training_step(self, batch, batch_idx) :
        loss, out, y = self._shared_step(batch, batch_idx)
        self.train_auc.update(out, y)
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx) :
        loss, out, y = self._shared_step(batch, batch_idx)
        self.valid_auc.update(out, y)
        self.log("valid_loss", loss.item())

    def test_step(self, batch, batch_idx) :
        loss, out, y = self._shared_step(batch, batch_idx)
        self.test_preds.extend(out.tolist())
        self.test_reals.extend(y.tolist())
        self.test_auc.update(out, y)
        self.log("test_loss", loss.item())

    def on_training_epoch_end(self) :
        self.log("train_auc", self.train_auc.compute())

    def on_validation_epoch_end(self) :
        self.log("valid_auc", self.valid_auc.compute())

    def on_test_epoch_end(self) :
        self.log("test_auc", self.test_auc.compute())

class EpitopeDataset(Dataset) :
    def __init__(self, X, mask, y) :
        self.X = torch.tensor(X, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long).bool()
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        return self.X[idx], self.mask[idx], self.y[idx]

def pad(seq, max_padded_length) :
    return [0]+seq+(max_padded_length-len(seq)+1)*[0]

def tokenize(lm, seqs, ids, max_padded_length) :
    seq_with_id = list(zip(ids, seqs))
    seq_with_id.append(("dummy", "<mask>"*max_padded_length))
    _, _, batch_tokens = lm.alphabet.get_batch_converter()(seq_with_id)
    return batch_tokens[:-1]

def process_data_file(path, lm, max_padded_length) :
    with open(path, "r") as f :
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        f.close()

    n_lines = len(lines)

    seqs = [l.upper() for l in lines[1::2]]
    ids = [l[1:] for l in lines[::2]]
    seqs_tokenized = tokenize(lm, seqs, ids, max_padded_length)
    masks = [pad([1]*len(seq), max_padded_length) for seq in seqs]
    epitope_residues = [[1 if c.isupper() else 0 for c in l] for l in lines[1::2]]
    epitope_residues_padded = [pad(epr, max_padded_length) for epr in epitope_residues]

    return seqs_tokenized, masks, epitope_residues_padded

def train_kfold(
    train_path: str,
    test_path: str,
    finetune: bool = False,
    k: int = 5,
    max_padded_length: int = 1000,
    batch_size: int = 16,
) :
    best_model_checkpoints = []

    wandb_logger = WandbLogger(
        project="epitope",
        log_model="all"
    )

    lm = LM("esm2_t30_150M_UR50D", 30, 640, finetune=finetune)

    train_dataset = EpitopeDataset(*process_data_file(train_path, lm, max_padded_length))
    test_dataset = EpitopeDataset(*process_data_file(test_path, lm, max_padded_length))
    test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=batch_size, shuffle=False)

    kf = KFold(n_splits=k, random_state=13, shuffle=True)
    n_samples = len(train_dataset)
    fold_cnt = 1

    preds = []
    reals = []



    for train_idx, dev_idx in kf.split(range(n_samples)) :
        train_dataloader = DataLoader(Subset(train_dataset, train_idx), num_workers=8, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(Subset(train_dataset, dev_idx), num_workers=8, batch_size=batch_size, shuffle=False)

        model = EpitopeModel(
            lm=lm,
            hidden_dim=512,
            dropout=0.1,
        )

        lit_module = LitModule(
            criterion=nn.BCEWithLogitsLoss(),
            lr=0.001,
            model=model,
        )

        ModelSummary(lit_module, -1)

        auc_callback = ModelCheckpoint(
            monitor="valid_auc",
            dirpath="/data/scratch/aronto/epitope_clean/models/checkpoints/",
            mode="max",
        )

        tr = pl.Trainer(
            logger=wandb_logger,
            callbacks=[auc_callback],
            accelerator="gpu",
            strategy="ddp",
            max_epochs=1,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            gpus=-1,
            gradient_clip_val=0.5,
        )

        tr.fit(lit_module, train_dataloader, dev_dataloader)
        best_model_checkpoints.append(auc_callback.best_model_path)

        #best_model = LitModule.load_from_checkpoint(auc_callback.best_model_path)
        #test_tr = pl.Trainer(
        #    accelerator="gpu",
        #    strategy="ddp",
        #    gpus=-1,
        #    enable_checkpointing=False,
        #)
        #test_tr.test(best_model, test_dataloader)

        #preds.extend(best_model.test_preds)
        #reals.extend(best_model.test_reals)

        #fold_cnt += 1

    #print(roc_auc_score(reals, preds))

    return best_model_checkpoints


if __name__ == "__main__" :
    l = train_kfold(
        train_path="/data/scratch/aronto/epitope_clean/data/BP3C50ID/train.fasta",
        test_path="/data/scratch/aronto/epitope_clean/data/BP3C50ID/test.fasta",
        finetune=False,
        k=2,
        max_padded_length=950,
        batch_size=2,
    )

    model = LitModule.load_from_checkpoint(l[0])











