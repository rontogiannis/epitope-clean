import sys
import os
import json
import argparse
import torch
import esm
import math
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import AUROC

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.loggers import WandbLogger

from egnn_pytorch import EGNN_Network, EGNN

from typing import Callable, Union
from operator import add

from sklearn.neighbors import NearestNeighbors

TRAIN = "/data/scratch/aronto/epitope_clean/data/GraphBepi/train.fasta"
IEDB = "/data/scratch/aronto/epitope_clean/data/IEDB/IEDB_reduced.fasta"
TEST = "/data/scratch/aronto/epitope_clean/data/GraphBepi/test.fasta"

class EpitopeDataset(Dataset) :
    def __init__(self, X, mask, y, coord, rho, iedb_emb) :
        self.X = torch.tensor(X, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long).bool()
        self.y = torch.tensor(y, dtype=torch.long)
        self.coord = torch.tensor(coord, dtype=torch.float)
        self.rho = torch.tensor(rho, dtype=torch.float)
        self.iedb_emb = torch.tensor(iedb_emb, dtype=torch.float)

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        return self.X[idx], self.mask[idx], self.y[idx], self.coord[idx], self.rho[idx], self.iedb_emb[idx]

class EGNNModule(nn.Module) :
    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        egnn_dim: int,
        egnn_nn: int,
    ) :
        super().__init__()

        self.egnn = EGNN(
            dim=embedding_dim,
            edge_dim=0,
            m_dim=egnn_dim,
            fourier_features=0,
            num_nearest_neighbors=egnn_nn,
            dropout=dropout,
            norm_feats=False,
            norm_coors=True,
            update_feats=True,
            update_coors=False,
            only_sparse_neighbors=False,
            valid_radius=float("inf"),
            m_pool_method="mean",
            soft_edges=True,
            coor_weights_clamp_value=None,
        )

    def forward(self, params) :
        emb, coors, mask = params
        out, coors_upd = self.egnn(emb, coors, mask=mask)
        return out, coors_upd, mask

class EpitopePredictionModel(nn.Module) :
    def __init__(
        self,
        esm_model_name: str,
        esm_layer_cnt: int,
        use_egnn: bool,
        use_rho: bool,
        esm_dim: int,
        egnn_dim: int,
        egnn_nn: int,
        egnn_layers: int,
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

        embedding_dim = esm_dim + 2 + (5 if use_rho else 0) # +2 for the IEDB embeddings +5 for rho

        # flags
        self.use_egnn = use_egnn
        self.use_rho = use_rho

        # Multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim//2, 1),
        )

        # Equivariant GNN
        self.egnn = nn.Sequential(
            *[EGNNModule(
                embedding_dim=embedding_dim,
                dropout=dropout,
                egnn_dim=egnn_dim,
                egnn_nn=egnn_nn,
            ) for layer in range(egnn_layers)]
        )

    def forward(self, X, mask, coors, rho, iedb_emb) :
        with torch.no_grad() :
            dct = self.esm_model(X, repr_layers=[self.ll_idx], return_contacts=False)
            emb = dct["representations"][self.ll_idx]
        emb = torch.cat((emb, iedb_emb), 2)
        emb = torch.cat((emb, rho), 2) if self.use_rho else emb
        emb = self.egnn((emb, coors, mask))[0] if self.use_egnn else emb
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
        X, mask, y, coors, rho, iedb_emb = batch
        out = self.model(X, mask, coors, rho, iedb_emb).squeeze(-1)

        # evaluation metrics
        top_idx = torch.argmax(out, dim=-1)
        top_k_idx = torch.topk(out, self.k, dim=-1).indices
        top_y = torch.gather(y, -1, top_idx.unsqueeze(-1)).squeeze(-1)
        top_y_k = torch.max(torch.gather(y, -1, top_k_idx), dim=-1).values

        # loss calculation
        mask = mask.flatten()
        out = out.flatten()
        out = torch.masked_select(out, mask)
        y = y.flatten()
        y = torch.masked_select(y, mask)
        loss = self.criterion(out, y.float())

        if update_auc :
            self.auc.update(out, y)

        if update_top_metric :
            self.yes += torch.sum(top_y)
            self.yes_k += torch.sum(top_y_k)
            self.all += top_y.shape[0]

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


def pad(seq, max_pad, empty=0) :
    return [empty]+seq+(max_pad-len(seq)+1)*[empty]

def tokenize(tokenizer, seqs, ids, max_pad) :
    seq_with_id = list(zip(ids, seqs)) + [("dummy", "<mask>"*max_pad)]
    _, _, batch_tokens = tokenizer.get_batch_converter()(seq_with_id)
    return batch_tokens[:-1]

COORDS = "/data/scratch/aronto/epitope_clean/data/GraphBepi/coords.json"
RHO = "/data/scratch/aronto/epitope_clean/data/GraphBepi/Rhos/rho_{}.json"

# utility functions for calculating the approximation of residue depth
def sq_norm(xi, xj) :
    a = xi[0]-xj[0]
    b = xi[1]-xj[1]
    c = xi[2]-xj[2]
    return a*a + b*b + c*c

def norm(xi) :
    a = xi[0]
    b = xi[1]
    c = xi[2]
    return math.sqrt(a*a + b*b + c*c)

def sum_vecs(vecs) :
    chi, psi, zed = 0, 0, 0
    for vec in vecs :
        chi += vec[0]
        psi += vec[1]
        zed += vec[2]
    return chi, psi, zed

def m_diff(m, xi, xj) :
    a = xi[0]-xj[0]
    b = xi[1]-xj[1]
    c = xi[2]-xj[2]
    return [m*a, m*b, m*c]

def residue_depth(coord, pdb_id, lambdas, kappa=10) :
    if os.path.isfile(RHO.format(pdb_id)) :
        with open(RHO.format(pdb_id), "r") as f :
            rho = json.load(f)
            f.close()
        return rho

    n = len(coord)
    ls = len(lambdas)

    _, NN = NearestNeighbors(n_neighbors=kappa+1, algorithm="ball_tree").fit(coord).kneighbors(coord)

    pairwise = [[[math.exp(-sq_norm(coord[i], coord[NN[i][j]])/lam)
                    for j in range(1,kappa+1)]
                    for i in range(n)]
                    for lam in lambdas]

    sums = [[sum(pairwise[l][i])
                for i in range(n)]
                for l in range(ls)]

    w = [[[pairwise[l][i][j]/sums[l][i]
            for j in range(kappa)]
            for i in range(n)]
            for l in range(ls)]

    numer = [[norm(sum_vecs([m_diff(w[l][i][j-1], coord[i], coord[NN[i][j]])
                for j in range(1,kappa+1)]))
                for i in range(n)]
                for l in range(ls)]

    denom = [[sum([w[l][i][j-1]*norm(m_diff(1, coord[i], coord[NN[i][j]]))
                for j in range(1,kappa+1)])
                for i in range(n)]
                for l in range(ls)]

    rho = [[numer[l][i]/denom[l][i]
            for l in range(ls)]
            for i in range(n)]

    with open(RHO.format(pdb_id), "w") as f :
        json.dump(rho, f, indent=4)
        f.close()

    return rho

def process_data_file(path, tokenizer, max_pad, is_iedb) :
    with open(path, "r") as f :
        lines = [l.strip() for l in f.readlines()]
        f.close()

    with open(COORDS, "r") as f :
        coor_dict = json.load(f)
        f.close()

    n = len(lines)
    seqs = [l.upper() for l in lines[1::2]]
    ids = [l[1:] for l in lines[::2]]

    # amino-acid tokens
    seqs_tok = tokenize(tokenizer, seqs, ids, max_pad).tolist()

    # masks
    masks = [pad([1]*len(seq), max_pad) for seq in seqs]

    # labels
    ep_resi = [[int(c.isupper()) for c in l] for l in lines[1::2]]
    ep_resi_pad = [pad(epr, max_pad) for epr in ep_resi]

    # IEDB flag-embeddings
    iedb_emb_single = [0, 0]
    iedb_emb_single[is_iedb] = 1
    iedb_emb = [[iedb_emb_single for j in range(max_pad+2)] for i in range(n)]

    # coordinate embeddings
    coors_unpad = [coor_dict[l[1:]] for l in lines[::2]]
    coors = [pad(coor, max_pad, empty=[0,0,0]) for coor in coors_unpad]

    # surface features (residue depth)
    rho_unpad = [residue_depth(coord, pdb_id[1:], [1., 2., 5., 10., 30.]) for coord, pdb_id in zip(coors_unpad, lines[::2])]
    rho = [pad(rd, max_pad, empty=[0,0,0,0,0]) for rd in rho_unpad]

    return [seqs_tok, masks, ep_resi_pad, coors, rho, iedb_emb]

MAX_PAD = 950
BATCH_SIZE = 4
NUM_WORKERS = 8
EPOCHS = 50
CHECKPOINTS = "/data/scratch/aronto/epitope_clean/models/checkpoints/"

def train(model, include_iedb) :
    auc_callback = ModelCheckpoint(
        monitor="validation/auc",
        dirpath=CHECKPOINTS,
        filename="best",
        auto_insert_metric_name=False,
        mode="max",
    )

    ModelSummary(model)

    raw_train = process_data_file(TRAIN, model.model.esm_model.alphabet, MAX_PAD, is_iedb=0)
    raw_iedb = process_data_file(IEDB, model.model.esm_model.alphabet, MAX_PAD, is_iedb=1) if include_iedb else None # TODO: currently we don't have the IEDB structures
    raw_concat = list(map(add, raw_train, raw_iedb)) if include_iedb else raw_train

    train_dataset = EpitopeDataset(*raw_concat)
    train_size = len(train_dataset)
    train_split, dev_split = random_split(
        dataset=train_dataset,
        lengths=[train_size-train_size//10, train_size//10],
    )

    train_loader = DataLoader(train_split, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_split, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[auc_callback],
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        log_every_n_steps=10,
        max_epochs=EPOCHS,
    )

    trainer.fit(model, train_loader, dev_loader)

    return auc_callback.best_model_path

def setup_cmd() :
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", help="test the last checkpointed model", action="store_true")
    parser.add_argument("--train", help="train a model (prioritized over testing, cannot do both at the same time)", action="store_true")
    parser.add_argument("--egnn", help="include equivariant GNN as part of the architecture", action="store_true")
    parser.add_argument("--rho", help="include residue depth as part of the embeddings", action="store_true")
    parser.add_argument("--iedb", help="include IEDB data in training", action="store_true")
    parser.add_argument("--seed", help="set the seed (default 13)", type=int, default=137)

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
    args = setup_cmd()

    pl.seed_everything(args["seed"])

    esm_models = {
        "3B": {"name": "esm2_t36_3B_UR50D", "layer_cnt": 36, "dim": 2560},
        "150M": {"name": "esm2_t30_150M_UR50D", "layer_cnt": 30, "dim": 640},
        "35M": {"name": "esm2_t12_35M_UR50D", "layer_cnt": 12, "dim": 480},
        "8M": {"name": "esm2_t6_8M_UR50D", "layer_cnt": 6, "dim": 320},
    }

    if args["train"] :
        model = EpitopeLitModule(
            criterion=nn.BCEWithLogitsLoss(),
            lr=0.001,
            esm_model_name=esm_models["150M"]["name"],
            esm_layer_cnt=esm_models["150M"]["layer_cnt"],
            esm_dim=esm_models["150M"]["dim"],
            use_egnn=args["egnn"],
            use_rho=args["rho"],
            egnn_dim=256,
            egnn_nn=10,
            egnn_layers=2,
            mlp_hidden_dim=256,
            dropout=0.1,
        )

        checkpoint_path = train(model, args["iedb"])
        print(f"Best checkpoint saved at {checkpoint_path}")
    elif args["test"] :
        test(CHECKPOINTS+"best.ckpt")





