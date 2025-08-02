# app/main.py
import numpy as np
import json
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# --- embeddings ---
U = np.load("../checkpoints/U.npy")   # (500, 64)
V = np.load("../checkpoints/V.npy")   # (1000, 64)

# --- hashtags ---
with open("../data/trending_videos.json", "r", encoding="utf-8") as f:
    raw = json.load(f)["collector"]

hashtags = []
for v in raw:
    tag_list = [f"#{h['name']}" for h in v.get("hashtags", [])]
    hashtags.append(" ".join(tag_list[:3]))   # one string per video

# --- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend(user_id: int = Query(..., ge=0, lt=len(U))):
    scores = U[user_id] @ V.T
    top_idx = np.argsort(scores)[-20:][::-1]

    results = []
    for vid in top_idx:
        pct = 100 * (len(V) - np.searchsorted(np.sort(scores), scores[vid])) / len(V)
        results.append({
            "video_id": int(vid),
            "score": float(scores[vid]),
            "percentile": float(pct),
            "hashtags": hashtags[vid] if hashtags[vid] else ""
        })
    return results

@app.post("/interact")
def interact(payload: dict):
    print(payload)  # stub log
    return {"status": "ok"}