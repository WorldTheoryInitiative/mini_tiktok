# train.py
import torch, time, os, json
from torch.utils.data import Dataset, DataLoader
from model import Recommender, triplet_loss
import pandas as pd
import numpy as np

# ------------- config -------------
N_USERS  = 500
N_VIDEOS = 1000
EMB_DIM  = 64
BATCH    = 512
EPOCHS   = 10
LR       = 1e-3
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPLET_FILE = os.path.join(THIS_DIR, "triplets.csv")
MODEL_DIR    = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)
# ----------------------------------

# 1. read the triplets
df = pd.read_csv(TRIPLET_FILE)
triplets = df[["user_idx", "pos_video_idx", "neg_video_idx"]].values.astype(np.int64)

class TripletDS(Dataset):
    def __len__(self): return len(triplets)
    def __getitem__(self, idx):
        u, p, n = triplets[idx]
        return torch.tensor(u), torch.tensor(p), torch.tensor(n)

loader = DataLoader(TripletDS(), batch_size=BATCH, shuffle=True)

# 2. model + optimiser
model = Recommender(N_USERS, N_VIDEOS, EMB_DIM)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

# 3. train loop
for epoch in range(1, EPOCHS+1):
    epoch_loss = 0.0
    for u, p, n in loader:
        opt.zero_grad()
        pos = model(u, p)
        neg = model(u, n)
        loss = triplet_loss(pos, neg)
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * len(u)
    epoch_loss /= len(triplets)
    print(f"epoch {epoch:02d}  loss={epoch_loss:.4f}")

# 4. save the learned notebooks
U = model.user_emb.weight.detach().cpu().numpy()
V = model.video_emb.weight.detach().cpu().numpy()
np.save(os.path.join(MODEL_DIR, "U.npy"), U)
np.save(os.path.join(MODEL_DIR, "V.npy"), V)
print("Saved U.npy and V.npy")