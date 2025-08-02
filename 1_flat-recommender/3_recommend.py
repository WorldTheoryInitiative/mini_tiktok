#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# 1) load the two matrices we already built
# ------------------------------------------------------------------
VIDEOS = np.load("output/videos.npy")   # (1000, 2348)
USERS  = np.load("output/users.npy")    # (500,  2348)

# sanity check
assert VIDEOS.ndim == 2 and USERS.ndim == 2
N_VIDEOS, DIM = VIDEOS.shape
N_USERS,  _   = USERS.shape
assert DIM == USERS.shape[1]

# ------------------------------------------------------------------
# 2) helper: top-k for one user
# ------------------------------------------------------------------
def recommend(user_id: int, k: int = 20):
    if user_id < 0 or user_id >= N_USERS:
        raise ValueError(f"user_id must be in [0, {N_USERS-1}]")

    user_vec = USERS[user_id].reshape(1, -1)        # 1×2348
    sims = cosine_similarity(user_vec, VIDEOS)[0]   # 1×1000 → flat array
    top_idx = np.argsort(sims)[::-1][:k]            # indices of highest scores
    return top_idx.tolist()                         # these are the video IDs

# ------------------------------------------------------------------
# 3) command-line entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, required=True,
                        help="User index (0..{})".format(N_USERS-1))
    parser.add_argument("--k", type=int, default=10,
                        help="Number of videos to return")
    args = parser.parse_args()

    ids = recommend(args.user_id, args.k)
    print(" ".join(map(str, ids)))   # space-separated list