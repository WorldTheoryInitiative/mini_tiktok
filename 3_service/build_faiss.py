# build_faiss.py
import numpy as np
import faiss

V = np.load("checkpoints/V.npy").astype('float32')   # (1000, 64)

# IVF-PQ with 1 cluster (tiny data) + 8-byte codes
index = faiss.IndexIVFPQ(
    faiss.IndexFlatL2(64),   # coarse quantiser
    64,                      # dimension
    1,                       # nlist (clusters)
    8,                       # m (bytes per vector)
    8                        # bits per sub-vector
)
index.train(V)              # required even with 1 cluster
index.add(V)
faiss.write_index(index, "index.faiss")
print("index.faiss saved")