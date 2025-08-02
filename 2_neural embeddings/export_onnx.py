# export_onnx.py
import torch
import numpy as np
from model import Recommender   # same tiny class

N_USERS  = 500
N_VIDEOS = 1000
EMB_DIM  = 64

# fresh model
model = Recommender(N_USERS, N_VIDEOS, EMB_DIM)

# load the trained weights from numpy
U = torch.from_numpy(np.load("../checkpoints/U.npy"))
V = torch.from_numpy(np.load("../checkpoints/V.npy"))
model.user_emb.weight.data  = U
model.video_emb.weight.data = V
model.eval()

# dummy inputs
user_idx  = torch.tensor([0], dtype=torch.long)
video_idx = torch.tensor([0], dtype=torch.long)

torch.onnx.export(
    model,
    (user_idx, video_idx),
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["user_idx", "video_idx"],
    output_names=["score"],
    dynamic_axes={"user_idx": {0: "batch"},
                  "video_idx": {0: "batch"},
                  "score": {0: "batch"}}
)

print("model.onnx saved ✔")