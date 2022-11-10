import torch
import esm
import biotite.structure.io as bsio
import matplotlib.pyplot as plt
import numpy as np
import re

from tqdm.auto import tqdm

print("Loading ESMFold.")

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

print("Loaded.")

path = "BP3C50ID/test.fasta"
out = "BP3C50ID/dump/"

with open(f"{path}", "r") as f :
    lines_with_sn = f.readlines()
    lines = [l.strip() for l in lines_with_sn]

pLDDTs = []

for i in tqdm(range(0, len(lines), 2)) :
    pdb_chain_id = lines[i][1:]
    sequence = lines[i+1].upper()

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(f"{out}{pdb_chain_id}.pdb", "w") as f :
        f.write(output)

    struct = bsio.load_structure(f"{out}{pdb_chain_id}.pdb", extra_fields=["b_factor"])
    pLDDTs.append(struct.b_factor.mean())

mu = np.mean(pLDDTs)
sigma = np.std(pLDDTs)

n, bins, patches = plt.hist(pLDDTs, 5, density=True, facecolor='g', alpha=0.75)

plt.xlabel('pLDDT Score')
plt.ylabel('Number of Observations')
plt.title('Certainty of Predicted Structures')
plt.text(60, .025, fr'$\mu=${mu}, $\sigma=${sigma}')
plt.xlim(0, 100)
plt.ylim(0, len(pLDDTs))
plt.grid(True)
plt.savefig("hist.png")



