#!/usr/bin/env python3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- config ----------------
K          = 10
U_FILE     = "checkpoints/U.npy"
V_FILE     = "checkpoints/V.npy"
OLD_VIDEO  = "1_flat-recommender/output/videos.npy"   # 2348-D vectors
OLD_USER   = "1_flat-recommender/output/users.npy"    # 2348-D vectors
LIKES_FILE = "data/interactions.csv"    # same file we built
# ----------------------------------------

# 1. load data
U = np.load(U_FILE)          # (n_users, 64)
V = np.load(V_FILE)          # (n_videos, 64)
old_U = np.load(OLD_USER)    # (n_users, 2348)
old_V = np.load(OLD_VIDEO)   # (n_videos, 2348)

import pandas as pd
likes = pd.read_csv(LIKES_FILE)

# build dict: user_id -> set of video_ids he *liked*
user_likes = likes.groupby("user_id")["video_id"].apply(set).to_dict()

n_users = len(user_likes)
recall_old = 0
recall_new = 0

# 2. helper
def recall_at_k(u_vec, V_mat, liked_set, k):
    sims = cosine_similarity(u_vec.reshape(1, -1), V_mat)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    hits = len(set(top_k) & liked_set)
    return hits / min(k, len(liked_set))

# 3. evaluate every user
for uid in range(n_users):
    liked = user_likes.get(uid, set())
    if not liked:
        continue
    # old vectors (2348-D)
    recall_old += recall_at_k(old_U[uid], old_V, liked, K)
    # new vectors (64-D)
    recall_new += recall_at_k(U[uid],     V,     liked, K)

recall_old /= n_users
recall_new /= n_users

print(f"Recall@{K}  baseline={recall_old:.3f}   neural={recall_new:.3f}")