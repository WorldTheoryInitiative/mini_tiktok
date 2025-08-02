TikTok Mini-Recommender – 3-Part Demo  
(Flat → Neural → Web App)

──────────────────────────  
Part 1 – Flat Baseline (Week 3-7)
1. Data  
   • 1 000 TikTok rows (description + hashtags).  
   • 6 k synthetic “likes” linking 500 users ↔ 1 000 videos.

2. Video vectors (2348-D)  
   • TF-IDF on hashtags → 2220-D sparse (scikit-learn).  
   • Word2Vec (128-D) on descriptions → dense (gensim).  
   • Concatenate → videos.npy (1000×2348).

3. User vectors (2348-D)  
   • Weighted average of liked-video vectors → users.npy (500×2348).

4. Retrieval  
   • `python recommend.py --user_id 1234` prints top-20 video IDs via cosine similarity.

──────────────────────────  
Part 2 – Neural Embeddings (Week 8-14)
1. Triplets  
   • Built 12 k (user, pos_video, neg_video) triplets from the 6 k likes with random negatives.

2. Model  
   • Two 64-D embedding tables (users & videos).  
   • Logistic loss: –log σ(u·v⁺) – log σ(–u·v⁻).  
   • Trained 10 epochs with PyTorch + Adam on Colab GPU.

3. Results  
   • Recall@20 improved from 0.28 (flat) → 0.47 (neural).  
   • Saved final embeddings as U.npy (500×64) and V.npy (1000×64).

4. Artifacts  
   • train.py, model.py, demo.ipynb, wandb curve screenshot.

──────────────────────────  
Part 3 – Web App (Week 15-26)
1. Backend  
   • ONNX-exported neural model served via FastAPI.  
   • Embeddings live in Redis (video_id → 64-float16).  
   • Faiss IVF-PQ index for < 50 ms nearest-neighbor on CPU.

2. Frontend  
   • React single-page app: infinite scroll, like/dislike buttons.  
   • Calls `/recommend?user_id=42` and `/interact` endpoints.

3. Deployment  
   • Docker-compose bundles FastAPI, Redis, Faiss, React.  
   • Live demo at https://my-mini-tiktok.fly.dev (free tier).

Quick start
Local backend only:
docker-compose up --build
Then open http://localhost:3000