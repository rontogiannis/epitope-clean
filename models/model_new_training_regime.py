import sys
import argparse
import torch
import esm
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import AUROC

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger

from typing import Callable, Union
from operator import add

TRAIN = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/train.fasta"
IEDB = "/data/scratch/aronto/epitope_clean/data/IEDB/IEDB_reduced.fasta"
TEST = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/test.fasta"

class EpitopeDataset(Dataset) :
    def __init__(self, X, mask, y, iedb_emb) :
        self.X = torch.tensor(X, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long).bool()
        self.y = torch.tensor(y, dtype=torch.long)
        self.iedb_emb = torch.tensor(iedb_emb, dtype=torch.float)

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        return self.X[idx], self.mask[idx], self.y[idx], self.iedb_emb[idx]

class EpitopePredictionModel(nn.Module) :
    def __init__(
        self,
        esm_model_name: str,
        esm_layer_cnt: int,
        esm_dim: int,
        mlp_hidden_dim: int,
        dropout: float,
    ) :
        super().__init__()

        # Load ESM language model
        try :
            esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model_name)
        except Exception :
            bar = getattr(esm.pretrained, esm_model_name)
            esm_model, alphabet = bar()

        self.esm_model = esm_model
        self.alphabet = alphabet
        self.ll_idx = esm_layer_cnt

        for param in self.esm_model.parameters() :
            param.requires_grad = False # We are not finetuning

        # Multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(esm_dim+2, mlp_hidden_dim), # +2 for the IEDB embeddings
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim//2, 1),
        )

    def forward(self, X, iedb_emb) :
        with torch.no_grad() :
            dct = self.esm_model(X, repr_layers=[self.ll_idx], return_contacts=False)
            emb = dct["representations"][self.ll_idx]
        emb = torch.cat((emb, iedb_emb), 2)
        out = self.mlp(emb)
        return out

class EpitopeLitModule(pl.LightningModule) :
    def __init__(
        self,
        criterion: Callable,
        lr: float,
        k: int = 10,
        **kwargs,
    ) :
        super().__init__()

        self.save_hyperparameters()

        self.k = k
        self.criterion = criterion
        self.model = EpitopePredictionModel(**kwargs)

        self.auc = AUROC()

        self.yes = 0
        self.yes_k = 0
        self.all = 0

    def configure_optimizers(self) :
        return torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.hparams.lr)

    def _shared_step(self, batch, batch_idx, update_auc=True, update_top_metric=True) :
        X, mask, y, iedb_emb = batch

        out = self.model(X, iedb_emb).squeeze(-1)

        # loss
        out_top_k_obj = torch.topk(out, self.k, dim=-1)
        y_top_k = torch.gather(y, -1, out_top_k_obj.indices)
        loss = self.criterion(out_top_k_obj.values, y_top_k.float())

        # other metrics
        out_top_idx = out_top_k_obj.indices[:,:1]
        y_top_k_pooled = torch.max(y_top_k, dim=-1).values
        y_top = torch.gather(y, -1, out_top_idx) #.squeeze(-1)

        #maxo = torch.max(out, dim=-1)
        #top_idx = maxo.indices # torch.argmax(out, dim=-1)
        #top_out = maxo.values
        #top_k_idx = torch.topk(out, self.k, dim=-1).indices
        #top_y = torch.gather(y, -1, top_idx.unsqueeze(-1)).squeeze(-1)
        #top_y_k = torch.max(torch.gather(y, -1, top_k_idx), dim=-1).values
        mask = mask.flatten()
        out = out.flatten()
        out = torch.masked_select(out, mask)
        y = y.flatten()
        y = torch.masked_select(y, mask)
        #loss = self.criterion(top_out, top_y.float()) # (out, y.float())

        if update_auc :
            self.auc.update(out, y)

        if update_top_metric :
            self.yes += torch.sum(y_top)
            self.yes_k += torch.sum(y_top_k_pooled)
            self.all += y_top.shape[0]

        return loss

    def on_validation_start(self) :
        self.auc.reset()
        self.yes = 0
        self.yes_k = 0
        self.all = 0

    def on_test_start(self) :
        self.auc.reset()
        self.yes = 0
        self.yes_k = 0
        self.all = 0

    def training_step(self, batch, batch_idx) :
        loss = self._shared_step(batch, batch_idx, update_auc=False)
        self.log("training/loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx) :
        self.log("validation/loss", self._shared_step(batch, batch_idx).item())

    def test_step(self, batch, batch_idx) :
        self.log("test/loss", self._shared_step(batch, batch_idx).item())

    def on_validation_epoch_end(self) :
        self.log("validation/auc", self.auc.compute())
        self.log("validation/top_acc", self.yes/self.all)
        self.log("validation/top_acc_k", self.yes_k/self.all)

    def on_test_epoch_end(self) :
        self.log("test/auc", self.auc.compute())
        self.log("test/top_acc", self.yes/self.all)
        self.log("test/top_acc_k", self.yes_k/self.all)


def pad(seq, max_pad) :
    return [0]+seq+(max_pad-len(seq)+1)*[0]

def tokenize(tokenizer, seqs, ids, max_pad) :
    seq_with_id = list(zip(ids, seqs)) + [("dummy", "<mask>"*max_pad)]
    _, _, batch_tokens = tokenizer.get_batch_converter()(seq_with_id)
    return batch_tokens[:-1]

def process_data_file(path, tokenizer, max_pad, is_iedb) :
    with open(path, "r") as f :
        lines = [l.strip() for l in f.readlines()]
        f.close()

    n = len(lines)
    seqs = [l.upper() for l in lines[1::2]]
    ids = [l[1:] for l in lines[::2]]
    seqs_tok = tokenize(tokenizer, seqs, ids, max_pad).tolist()
    masks = [pad([1]*len(seq), max_pad) for seq in seqs]
    ep_resi = [[int(c.isupper()) for c in l] for l in lines[1::2]]
    ep_resi_pad = [pad(epr, max_pad) for epr in ep_resi]
    iedb_emb_single = [0, 0]
    iedb_emb_single[is_iedb] = 1
    iedb_emb = [[iedb_emb_single for j in range(max_pad+2)] for i in range(n)]

    return [seqs_tok, masks, ep_resi_pad, iedb_emb]

MAX_PAD = 950
BATCH_SIZE = 3
NUM_WORKERS = 8
EPOCHS = 50
CHECKPOINTS = "/data/scratch/aronto/epitope_clean/models/checkpoints/"

def train(model, include_iedb) :
    metric_callback = ModelCheckpoint(
        monitor="validation/top_acc",
        dirpath=CHECKPOINTS,
        filename="best",
        auto_insert_metric_name=False,
        mode="max",
    )

    ModelSummary(model)

    raw_train = process_data_file(TRAIN, model.model.esm_model.alphabet, MAX_PAD, is_iedb=0)
    raw_iedb = process_data_file(IEDB, model.model.esm_model.alphabet, MAX_PAD, is_iedb=0)
    raw_concat = list(map(add, raw_train, raw_iedb)) if include_iedb else raw_train

    train_dataset = EpitopeDataset(*raw_concat)
    train_size = len(train_dataset)
    train_split, dev_split = random_split(
        dataset=train_dataset,
        lengths=[train_size-train_size//5, train_size//5],
    )

    train_loader = DataLoader(train_split, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_split, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[metric_callback],
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        log_every_n_steps=10,
        max_epochs=EPOCHS,
    )

    trainer.fit(model, train_loader, dev_loader)

    return metric_callback.best_model_path

def setup_cmd() :
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", help="to test the last checkpointed model", action="store_true")
    parser.add_argument("--train", help="to train a model (prioritized over testing, cannot do both at the same time)", action="store_true")
    parser.add_argument("--iedb", help="to include IEDB data in training", action="store_true")

    args = vars(parser.parse_args())

    return args

def test(path) :
    model = EpitopeLitModule.load_from_checkpoint(path, map_location="cpu")

    test_dataset = EpitopeDataset(*process_data_file(TEST, model.model.esm_model.alphabet, MAX_PAD, is_iedb=0))
    test_loader = DataLoader(test_dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)

    tester = pl.Trainer(
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    tester.test(model, test_loader)

if __name__ == "__main__" :
    pl.seed_everything(13)

    args = setup_cmd()

    if args["train"] :
        model = EpitopeLitModule(
            criterion=nn.BCEWithLogitsLoss(),
            lr=0.001,
            k=10,
            esm_model_name="esm2_t30_150M_UR50D",
            esm_layer_cnt=30,
            esm_dim=640,
            mlp_hidden_dim=256,
            dropout=0.1,
        )

        train(model, args["iedb"])
    elif args["test"] :
        test(CHECKPOINTS+"best.ckpt")





