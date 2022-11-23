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
import biotite.structure.io as bsio

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
from egnn_pytorch import EGNN_Network, EGNN
from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder, XTransformer
from se3_transformer_pytorch import SE3Transformer

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
test_path = "/data/scratch/aronto/epitope_clean/data/IEDB/IEDB_reduced_test.fasta" # "/data/scratch/aronto/epitope_clean/data/BP3C50ID/test.fasta"
test_pdbs = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/test-pred/"

ESM_MODELS = [
    ("esm2_t30_150M_UR50D", 30, 640),
    ("esm1b_t33_650M_UR50S", 33, 1280),
    ("esm2_t33_650M_UR50D", 33, 1280),
    ("esm2_t6_8M_UR50D", 6, 320),
]

esm_model_name, esm_model_layer_count, esm_embedding_dim = ESM_MODELS[0]

batch_size = 1
epochs = 30

kappa = 8
lambdas = [1., 2., 5., 10., 30.]
ls = len(lambdas)

# max padded length of sequence
max_padded_length = 34355 # 950 # 933 actual maximum length on training/test data, 343550 when testing on IEDB

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
        print(f" {self.X.shape=}")
        print(f" {self.mask.shape=}")
        print(f" {self.y.shape=}")
        print(f" {self.coord.shape=}")
        print(f" {self.rhos.shape=}")

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
def process_data_file(path, pdbs, take_all=False, struct_feats=False) :
    print(f"Processing dataset {path}")

    with open(path, "r") as f :
        lines_with_sn = f.readlines()
        lines = [l.strip() for l in lines_with_sn]
        f.close()

    sequences = [lines[i].upper() for i in range(1, len(lines), 2)]
    masks = [[0]+[1]*len(seq)+(max_padded_length-len(seq)+1)*[0] for seq in sequences]
    pdb_ids = [lines[i][1:].split("_")[0] if "_" in lines[i][1:] else lines[i][1:] for i in range(0, len(lines), 2)]
    chain_ids = [lines[i][1:].split("_")[1] if "_" in lines[i][1:] else "NA" for i in range(0, len(lines), 2)]
    epitope_residues = [[0]+[1 if c.isupper() else 0 for c in lines[i]]+(max_padded_length-len(lines[i])+1)*[0] for i in range(1, len(lines), 2)]

    # tokenize
    seq_with_id = [p for p in zip(pdb_ids, sequences)]
    seq_with_id.append(("dummy", "<mask>"*max_padded_length))
    _, _, batch_tokens = batch_converter(seq_with_id)
    batch_tokens = batch_tokens[:-1] # X

    if struct_feats :
        # get 3D structures
        coordinates = []
        rhos = []
        indx = 0
        indices = []

        print("Calculating 3D features")
        for pdb_id, chain_id, seq in tqdm(zip(pdb_ids, chain_ids, sequences)) :
            struct = bsio.load_structure(f"{pdbs}{pdb_id}_{chain_id}.pdb", extra_fields=["b_factor"])
            if struct.b_factor.mean() >= 80.0 or take_all :
                indices.append(indx)
            indx += 1

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

    if struct_feats :
        return np.array(batch_tokens.tolist())[indices], np.array(masks)[indices], np.array(epitope_residues)[indices], np.array(coordinates)[indices], np.array(rhos)[indices]
    else :
        return np.array(batch_tokens.tolist()), np.array(masks), np.array(epitope_residues), np.zeros_like(masks), np.zeros_like(masks)

X_train, mask_train, y_train, coord_train, rhos_train = process_data_file(train_path, train_pdbs, take_all=True, struct_feats=False)
X_test, mask_test, y_test, coord_test, rhos_test = process_data_file(test_path, test_pdbs, take_all=True, struct_feats=False)

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
        egnn_depth: int = 3, # TODO: for now this has no effect
        egnn_max_nn: int = 16,
        dropout: int = 0.2,
        finetune_lm: bool = False,
        use_rho: bool = False,
        use_egnn: bool = False,
        use_transformer: bool = False,
        use_se3_transformer: bool = False,
        n_heads: int = 2,
        transformer_dim: int = 512,
        transformer_out_dim: int = 256,
        transformer_depth: int = 2,
        transformer_n_heads: int = 4,
        se3_transformer_n_heads: int = 4,
        se3_transformer_depth: int = 2,
        se3_transformer_dim_head: int = 4,
        se3_transformer_num_degrees: int = 4,
        se3_transformer_valid_radius: int = 4
    ) :
        super().__init__()

        self.finetune_lm = finetune_lm
        self.use_rho = use_rho
        self.use_egnn = use_egnn
        self.use_se3_transformer = use_se3_transformer
        self.use_transformer = use_transformer
        self.embedding_dim = embedding_dim+(rho_dimension if use_rho else 0)

        if finetune_lm :
            self.esm_embedder = copy.deepcopy(esm_model) # TODO: i dont like using global variables, pls change that
        else :
            self.esm_embedder = esm_model

        #self.attn = nn.MultiheadAttention(
        #    embed_dim=self.embedding_dim,
        #    num_heads=n_heads,
        #    batch_first=True,
        #)

        if use_se3_transformer :
            # too much memory
            self.se3_transformer = SE3Transformer(
                dim = self.embedding_dim,
                heads = se3_transformer_n_heads,
                depth = se3_transformer_depth,
                dim_head = se3_transformer_dim_head,
                num_degrees = se3_transformer_num_degrees,
                num_neighbors = 4,
                valid_radius = se3_transformer_valid_radius
            )

        if use_transformer :
            self.transformer = ContinuousTransformerWrapper(
                dim_in = self.embedding_dim,
                dim_out = transformer_out_dim,
                max_seq_len = max_padded_length+2,
                attn_layers = Decoder(
                    dim = transformer_dim,
                    depth = transformer_depth,
                    heads = transformer_n_heads,
                    rotary_pos_emb = True,
                    attn_dropout = dropout,
                    ff_dropout = dropout,
                )
            )

        if use_egnn :
            # TODO: this is ugly, figure out a way to have a list of EGNN's that works with DataParallel
            self.egnn_1 = EGNN(
                dim=self.embedding_dim,            # input dimension
                edge_dim=0,                        # dimension of the edges, if exists, should be > 0
                m_dim=egnn_dim,                    # hidden model dimension
                fourier_features=0,                # number of fourier features for encoding of relative distance - defaults to none as in paper
                num_nearest_neighbors=egnn_max_nn, # cap the number of neighbors doing message passing by relative distance
                dropout=dropout,                   # dropout
                norm_feats=False,                  # whether to layernorm the features
                norm_coors=True,                   # whether to normalize the coordinates, using a strategy from the SE(3) Transformers paper
                update_feats=True,                 # whether to update features - you can build a layer that only updates one or the other
                update_coors=False,                # whether ot update coordinates
                only_sparse_neighbors=False,       # using this would only allow message passing along adjacent neighbors, using the adjacency matrix passed in
                valid_radius=float('inf'),         # the valid radius each node considers for message passing
                m_pool_method='mean',              # whether to mean or sum pool for output node representation
                soft_edges=False,                  # extra GLU on the edges, purportedly helps stabilize the network in updated version of the paper
                coor_weights_clamp_value=None      # clamping of the coordinate updates, again, for stabilization purposes
            )

            self.egnn_2 = EGNN(
                dim=self.embedding_dim,
                edge_dim=0,
                m_dim=egnn_dim,
                fourier_features=0,
                num_nearest_neighbors=egnn_max_nn,
                dropout=dropout,
                norm_feats=False,
                norm_coors=True,
                update_feats=True,
                update_coors=False,
                only_sparse_neighbors=False,
                valid_radius=float('inf'),
                m_pool_method='mean',
                soft_edges=False,
                coor_weights_clamp_value=None
            )


        self.linear = nn.Sequential(
            nn.Linear((transformer_out_dim if use_transformer else self.embedding_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
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
        if self.use_rho : embeddings = torch.cat((embeddings, rho), 2)
        if self.use_egnn :
            embeddings, _ = self.egnn_1(embeddings, coords, mask=mask)
            embeddings, _ = self.egnn_2(embeddings, coords, mask=mask)
        elif self.use_se3_transformer :
            embeddings = self.se3_transformer(embeddings, coords, mask)
        if self.use_transformer :
            embeddings = self.transformer(embeddings, mask=mask)
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
        hidden_dim=256,
        egnn_dim=512,
        egnn_depth=2,
        egnn_max_nn=8,
        dropout=0.2,
        finetune_lm=False,
        use_rho=False,
        use_egnn=False,
        use_se3_transformer=False,
        use_transformer=False,
        transformer_dim=512,
        transformer_depth=2,
        transformer_n_heads=4,
        transformer_out_dim=256,
    )).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

    # training loop
    print(f"> Training model for split {split_count}.")
    split_count += 1

    no_improvement = 0

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
            no_improvement = 0
        else :
            no_improvement += 1
            if no_improvement >= 10 :
                print("Early Stopping.")
                break

    # test
    print("> Running best model on test dataset.")
    best_model.eval()
    with torch.no_grad() :
        auc, _, test_loss, preds, reals = run(best_model, test_dataloader, epoch, False)
        all_predictions.extend(preds)
        all_reals.extend(reals)

    # TODO: add parameter saving for reproducability

print(f"AUC={roc_auc_score(all_reals, all_predictions)}")


