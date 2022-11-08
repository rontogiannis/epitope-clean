import os
import sys

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import copy
import esm
import csv

from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, DataParallel
from torch.utils.data import dataset, Dataset, DataLoader
from io import StringIO
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn.model_selection import KFold
from Bio.PDB.PDBParser import PDBParser
from sklearn.neighbors import NearestNeighbors
from egnn_pytorch import EGNN_Network

d3to1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

# make everything reproducable
torch.manual_seed(13)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# command-line argument processing
#TODO: allow custom command-line arguments
#TODO: allow loading pretrained model

train_path = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/train.fasta"
train_pdbs = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/train-pred/"
test_path = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/test.fasta"
test_pdbs = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/test-pred/"

ESM_MODELS = [
    ("esm2_t30_150M_UR50D", 30, 640),
    ("esm1b_t33_650M_UR50S", 33, 1280),
]

esm_model_name, esm_model_layer_count, esm_embedding_dim = ESM_MODELS[0]

batch_size = 16
epochs = 200

kappa = 16
lambdas = [1., 2., 5., 10., 30.]
ls = len(lambdas)

# max padded length of sequence
max_padded_length = 950 # 933 actual maximum length on training/test data

# load esm2
try :
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model_name)
except Exception :
    bar = getattr(esm.pretrained, esm_model_name)
    esm_model, alphabet = bar()

esm_model = esm_model.to(device)
batch_converter = alphabet.get_batch_converter()

class EpitopeDataset(Dataset) :
    def __init__(self, X, mask, y, coord, rhos) :
        self.X = torch.tensor(X, dtype=torch.long).to(device) # N x (max_padded_length+2)
        self.mask = torch.tensor(mask, dtype=torch.long).bool().to(device) # N x (max_padded_length+2)
        self.y = torch.tensor(y, dtype=torch.long).to(device) # N x (max_padded_length+2)
        self.coord = torch.tensor(coord, dtype=torch.float).to(device) # N x (max_padded_length+2) x 3
        self.rhos = torch.tensor(rhos, dtype=torch.float).to(device) # N x (max_padded_length+2) x |Lambdas|

        self.show_shapes()

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        return self.X[idx], self.mask[idx], self.y[idx], self.coord[idx], self.rhos[idx]

    def show_shapes(self) :
        print("Shape check:")
        print(f"> {self.X.shape=}")
        print(f"> {self.mask.shape=}")
        print(f"> {self.y.shape=}")
        print(f"> {self.coord.shape=}")
        print(f"> {self.rhos.shape=}")

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

# loads dataset and returns X, y to be fed to the DataLoader
def process_data_file(path, pdbs) :
    print(f"Processing dataset {path}")

    with open(path, "r") as f :
        lines_with_sn = f.readlines()
        lines = [l.strip() for l in lines_with_sn]
        f.close()

    sequences = [lines[i].upper() for i in range(1, len(lines), 2)]
    masks = [[0]+[1]*len(seq)+(max_padded_length-len(seq)+1)*[0] for seq in sequences]
    pdb_ids = [lines[i][1:].split("_")[0] for i in range(0, len(lines), 2)]
    chain_ids = [lines[i][1:].split("_")[1] for i in range(0, len(lines), 2)]
    epitope_residues = [[0]+[1 if c.isupper() else 0 for c in lines[i]]+(max_padded_length-len(lines[i])+1)*[0] for i in range(1, len(lines), 2)]

    # tokenize
    seq_with_id = [p for p in zip(pdb_ids, sequences)]
    seq_with_id.append(("dummy", "<mask>"*max_padded_length))
    _, _, batch_tokens = batch_converter(seq_with_id)
    batch_tokens = batch_tokens[:-1] # X

    # get 3D structures
    coordinates = []
    rhos = []

    print("Calculating 3D features")
    for pdb_id, chain_id, seq in tqdm(zip(pdb_ids, chain_ids, sequences)) :
        parser = PDBParser(QUIET=True)
        model = parser.get_structure(pdb_id, f"{pdbs}{pdb_id}_{chain_id}.pdb")[0]
        coord = []

        for residue in model["A"] :
            for atom in residue :
                if atom.get_name() == "CA" :
                    coord.append(atom.get_coord().tolist())

        # residue depth approximation
        n = len(coord)
        _, NN = NearestNeighbors(n_neighbors=kappa, algorithm="ball_tree").fit(coord).kneighbors(coord)
        pairwise = [[[math.exp(-sq_norm(coord[i], coord[NN[i][j]])/lam)
                        for j in range(kappa)]
                        for i in range(n)]
                        for lam in lambdas]
        sums = [[sum(pairwise[l][i])
                    for i in range(n)]
                    for l in range(ls)]
        w = [[[pairwise[l][i][j]/sums[l][i]
                for j in range(kappa)]
                for i in range(n)]
                for l in range(ls)]
        numer = [[norm(sum_vecs([m_diff(w[l][i][j], coord[i], coord[NN[i][j]])
                    for j in range(kappa)]))
                    for i in range(n)]
                    for l in range(ls)]
        denom = [[sum([w[l][i][j]*norm(m_diff(1, coord[i], coord[NN[i][j]]))
                    for j in range(kappa)])
                    for i in range(n)]
                    for l in range(ls)]
        rho = [[numer[l][i]/denom[l][i]
                for l in range(ls)]
                for i in range(n)]

        coord_padded = [[.0, .0, .0]]+coord+(max_padded_length-len(seq)+1)*[[.0, .0, .0]]
        rho_padded = [[.0]*ls]+rho+(max_padded_length-len(seq)+1)*[[.0]*ls]

        coordinates.append(coord_padded)
        rhos.append(rho_padded)

    return np.array(batch_tokens.tolist()), np.array(masks), np.array(epitope_residues), np.array(coordinates), np.array(rhos)

X_train, mask_train, y_train, coord_train, rhos_train = process_data_file(train_path, train_pdbs)
X_test, mask_test, y_test, coord_test, rhos_test = process_data_file(test_path, test_pdbs)

test_dataset = EpitopeDataset(X_test, mask_test, y_test, coord_test, rhos_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# model
class EpitopeModel(nn.Module) :
    def __init__(
        self,
        embedding_dim: int,
        rho_dimension: int = 5,
        hidden_dim: int = 256,
        egnn_dim: int = 128,
        egnn_depth: int = 3,
        egnn_max_nn: int = 16,
        dropout: int = 0.2,
        finetune_lm: bool = False,
        use_rho: bool = False,
        use_egnn: bool = False,
    ) :
        super().__init__()

        self.finetune_lm = finetune_lm
        self.use_rho = use_rho
        self.use_egnn = use_egnn

        self.esm_embedder = esm_model # TODO: i dont like using global variables, pls change that

        self.egnn = EGNN_Network(
            num_tokens=32,
            num_positions=max_padded_length+2,
            dim=egnn_dim,
            depth=egnn_depth,
            num_nearest_neighbors=egnn_max_nn,
            coor_weights_clamp_value=2.
        )

        self.linear = nn.Sequential(
            nn.Linear(embedding_dim+(rho_dimension if use_rho else 0)+(egnn_dim if use_egnn else 0), hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim//2, 1),
        )

        for param in self.esm_embedder.parameters() :
            param.requires_grad = finetune_lm

    def forward(self, X: Tensor, coords: Tensor, mask: Tensor, rho: Tensor) -> Tensor :
        if self.finetune_lm == False :
            with torch.no_grad() :
                embeddings = self.esm_embedder(X, repr_layers=[esm_model_layer_count], return_contacts=True)
        else :
            embeddings = self.esm_embedder(X, repr_layers=[esm_model_layer_count], return_contacts=True)

        embeddings = embeddings["representations"][esm_model_layer_count].to(device)
        egnn_embeddings, egnn_coords = self.egnn(X, coords, mask=mask)
        if self.use_rho : embeddings = torch.cat((embeddings, rho), 2)
        if self.use_egnn : embeddings = torch.cat((embeddings, egnn_embeddings), 2)
        output = self.linear(embeddings)

        return output

# run a single training/eval/test iteration
def run(model, run_loader, epoch, training=False) :
    total_loss = 0.0
    batch_cnt = 0
    tq = tqdm(run_loader)
    batch_total = len(run_loader)

    preds = []
    reals = []
    auc = 0.0

    for batch in tq :
        X, mask, y, coord, rhos = batch
        output = model(X, coord, mask, rhos)

        mask = mask.flatten()
        y = torch.masked_select(y.flatten(), mask)
        output = torch.masked_select(output.flatten(), mask)

        loss = criterion(output, y.float())

        preds.extend(output.cpu().detach().numpy().tolist())
        reals.extend(y.cpu().detach().numpy().tolist())

        if training :
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5)
            optimizer.step()

        total_loss += loss.item()
        batch_cnt += 1

        if batch_cnt == batch_total :
            auc = roc_auc_score(reals, preds)
            tq.set_description('Epoch #{}/{} | batch loss: {:0.3f}, avg loss: {:0.3f} | AUC: {:0.3f}'.format(epoch, epochs, loss.item(), total_loss/batch_cnt, auc))
        else :
            tq.set_description('Epoch #{}/{} | batch loss: {:0.3f}, avg loss: {:0.3f} | AUC: ?'.format(epoch, epochs, loss.item(), total_loss/batch_cnt))

        tq.refresh()

    return auc, total_loss, total_loss/batch_cnt, preds, reals

# 5-fold cross validation
print("Commencing 5-fold cross-validation.")
kf = KFold(n_splits=5,
           random_state=13,
           shuffle=True)

all_predictions = []
all_reals = []
split_count = 0

for train_indices, dev_indices in kf.split(X_train) :
    # split the data
    X_train_cv = X_train[train_indices]
    mask_train_cv = mask_train[train_indices]
    y_train_cv = y_train[train_indices]
    coord_train_cv = coord_train[train_indices]
    rhos_train_cv = rhos_train[train_indices]

    X_dev = X_train[dev_indices]
    mask_dev = mask_train[dev_indices]
    y_dev = y_train[dev_indices]
    coord_dev = coord_train[dev_indices]
    rhos_dev = rhos_train[dev_indices]

    # create dataloaders
    train_dataset = EpitopeDataset(X_train_cv, mask_train_cv, y_train_cv, coord_train_cv, rhos_train_cv)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = EpitopeDataset(X_dev, mask_dev, y_dev, coord_dev, rhos_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    # initialize best model
    best_model = None
    best_auc = 0.0

    # initialize model, scheduler, optimizer
    model = nn.DataParallel(EpitopeModel(
        embedding_dim=esm_embedding_dim,
        rho_dimension=ls,
        hidden_dim=512,
        egnn_dim=128,
        egnn_depth=4,
        egnn_max_nn=8,
        dropout=0.1,
        finetune_lm= False,
        use_rho=True,
        use_egnn=True,
    )).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=4.0) # not setting it lower cause the scheduler will reduce it on validation score plateau

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.75,
        patience=2,
        min_lr=0.001,
        verbose=True
    )

    # training loop
    print(f"> Training model for split {split_count}.")
    split_count += 1

    for epoch in range(1, epochs+1) :
        # train
        model.train()
        run(model, train_dataloader, epoch, True)

        # eval
        model.eval()
        with torch.no_grad() :
            auc, _, val_loss, _, _ = run(model, dev_dataloader, epoch, False)

        # update best_model
        if auc > best_auc :
            best_auc = auc
            best_model = copy.deepcopy(model)

        # update lr based on dev loss
        scheduler.step(val_loss)

    # test
    print("> > Running best model on test dataset.")
    best_model.eval()
    with torch.no_grad() :
        auc, _, test_loss, preds, reals = run(best_model, test_dataloader, epoch, False)
        all_predictions.extend(preds)
        all_reals.extend(reals)

    # TODO: add parameter saving for reproducability

print(f"AUC={roc_auc_score(all_reals, all_predictions)}")


