import torch
import esm

from tqdm.auto import tqdm

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

path = "BP3C50ID/train.fasta"
out = "BP3C50ID/train-pred/"

with open(f"{path}", "r") as f :
    lines_with_sn = f.readlines()
    lines = [l.strip() for l in lines_with_sn]

for i in tqdm(range(0, len(lines), 2)) :
    pdb_chain_id = lines[i][1:]
    sequence = lines[i+1].upper()

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(f"{out}{pdb_chain_id}.pdb", "w") as f:
        f.write(output)

