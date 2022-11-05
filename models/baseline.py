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

torch.manual_seed(13)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# command-line argument processing
#TODO: allow custom command-line arguments
#TODO: allow loading pretrained model

train_path = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/train.fasta"
train_pdbs = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/train/"
test_path = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/test.fasta"
test_pdbs = "/data/scratch/aronto/epitope_clean/data/BP3C50ID/test/"

esm_model_name = "esm2_t30_150M_UR50D"
esm_model_layer_count = 30
esm_embedding_dim = 640
batch_size = 16
epochs = 75

# max padded length of sequence
max_padded_length = 1000 # 933 actual maximum length on training/test data

# load esm2
try :
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model_name)
except Exception :
    bar = getattr(esm.pretrained, esm_model_name)
    esm_model, alphabet = bar()

esm_model = esm_model.to(device)
batch_converter = alphabet.get_batch_converter()

class EpitopeDataset(Dataset) :
    def __init__(self, X, mask, y) :
        self.X = torch.tensor(X, dtype=torch.long).to(device) # N x (max_padded_length+2)
        self.mask = torch.tensor(mask, dtype=torch.long).to(device) # N x (max_padded_length+2)
        self.y = torch.tensor(y, dtype=torch.long).to(device) # N x (max_padded_length+2)
        # self.show_shapes()

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        return self.X[idx], self.mask[idx], self.y[idx]

    def show_shapes(self) :
        print(self.X.shape)
        print(self.mask.shape)
        print(self.y.shape)

# loads dataset and returns X, y to be fed to the DataLoader
def process_data_file(path, pdbs) :
    print(f"Processing dataset {path}")

    with open(path, "r") as f :
        lines = f.readlines()
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

    return np.array(batch_tokens.tolist()), np.array(masks), np.array(epitope_residues)

X_train, mask_train, y_train = process_data_file(train_path, train_pdbs)
X_test, mask_test, y_test = process_data_file(test_path, test_pdbs)

test_dataset = EpitopeDataset(X_test, mask_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# model
class EpitopeModel(nn.Module) :
    def __init__(
        self,
        embedding_dim: int,
        finetune_lm: bool = False,
    ) :
        super().__init__()

        self.embedding_dim = embedding_dim
        self.finetune_lm = finetune_lm

        self.esm_embedder = esm_model # TODO: i dont like using global variables, pls change that
        self.linear = nn.Linear(embedding_dim, 1)

        for param in self.esm_embedder.parameters() :
            param.requires_grad = finetune_lm

    def forward(self, X: Tensor) -> Tensor :
        if self.finetune_lm == False :
            with torch.no_grad() :
                embeddings = self.esm_embedder(X, repr_layers=[esm_model_layer_count], return_contacts=True)
        else :
            embeddings = self.esm_embedder(X, repr_layers=[esm_model_layer_count], return_contacts=True)
        embeddings = embeddings["representations"][esm_model_layer_count].to(device)
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
    auc = 0.5

    for batch in tq :
        X, mask, y = batch
        output = model(X)

        mask = mask.flatten().bool()
        y = torch.masked_select(y.flatten(), mask)
        output = torch.masked_select(output.flatten(), mask)

        loss = criterion(output, y.float())

        preds.extend(output.cpu().detach().numpy().tolist())
        reals.extend(y.cpu().detach().numpy().tolist())

        if training :
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        total_loss += loss.item()
        batch_cnt += 1

        if batch_cnt == batch_total :
            auc = roc_auc_score(reals, preds)
            tq.set_description('Epoch #{} | batch loss: {:0.3f}, avg loss: {:0.3f} | AUC: {:0.3f}'.format(epoch, loss.item(), total_loss/batch_cnt, auc))
        else :
            tq.set_description('Epoch #{} | batch loss: {:0.3f}, avg loss: {:0.3f} | AUC: ?'.format(epoch, loss.item(), total_loss/batch_cnt))

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

    X_dev = X_train[dev_indices]
    mask_dev = mask_train[dev_indices]
    y_dev = y_train[dev_indices]

    # create dataloaders
    train_dataset = EpitopeDataset(X_train_cv, mask_train_cv, y_train_cv)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = EpitopeDataset(X_dev, mask_dev, y_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    # initialize best model
    best_model = None
    best_auc = 0.0

    # initialize model, scheduler, optimizer
    model = nn.DataParallel(EpitopeModel(
        embedding_dim=esm_embedding_dim,
        finetune_lm=False
    )).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01) # not setting it lower cause the scheduler will reduce it on validation score plateau

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.75,
        patience=2,
        min_lr=0,
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

print(f"AUC={roc_auc_score(all_reals, all_predictions)}")


