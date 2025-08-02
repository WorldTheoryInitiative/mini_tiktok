#!/usr/bin/env python3
import numpy as np
import pandas as pd

# -------------------- parameters --------------------
N_VIDEOS  = 1_000      # must match videos.npy
N_USERS   = 500        # can be any number you like
N_EVENTS  = 8_000      # total (user, video, likes) rows
OUT_FILE  = "interactions.csv"
SEED      = 42
# ----------------------------------------------------

rng = np.random.default_rng(SEED)

# raw integer ids
user_ids   = rng.integers(0, N_USERS,  N_EVENTS)
video_ids  = rng.integers(0, N_VIDEOS, N_EVENTS)
likes      = rng.integers(1, 10,      N_EVENTS)   # weight 1–9

interactions = pd.DataFrame({
    "user_id" : user_ids,
    "video_id": video_ids,
    "likes"   : likes
})

# optional: drop duplicate (user,video) pairs and keep first
interactions = interactions.drop_duplicates(subset=["user_id", "video_id"])

# save
interactions.to_csv(OUT_FILE, index=False)
print(f"Saved {len(interactions)} rows → {OUT_FILE}")