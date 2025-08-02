# model.py
import torch
import torch.nn as nn

class Recommender(nn.Module):
    def __init__(self, n_users: int, n_videos: int, dim: int = 64):
        super().__init__()
        # two notebooks with blank pages
        self.user_emb  = nn.Embedding(n_users,  dim)   # page per user
        self.video_emb = nn.Embedding(n_videos, dim)   # page per video

    def forward(self, user_idx, video_idx):
        # open the pages and read the numbers
        u = self.user_emb(user_idx)          # shape (batch, 64)
        v = self.video_emb(video_idx)        # shape (batch, 64)
        # multiply line-by-line and add → one score per pair
        return (u * v).sum(dim=1)            # shape (batch,)

# ------------------------------------------------------------------
# Loss on triplets (logistic / BPR-style)
# ------------------------------------------------------------------
def triplet_loss(pos_score, neg_score):
    # push pos_score above neg_score
    return torch.mean(torch.sigmoid(neg_score - pos_score))