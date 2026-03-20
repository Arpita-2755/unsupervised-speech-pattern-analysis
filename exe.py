import numpy as np
from joblib import load

# ---------- PATHS ----------
EMBEDDING_MODEL_PATH = r"C:\Users\wisdo\Documents\ML2 Project\model_artifacts\embedding_model.joblib"
UMAP_MODEL_PATH       = r"C:\Users\wisdo\Documents\ML2 Project\model_artifacts\umap_model.joblib"
KMEANS_PATH           = r"C:\Users\wisdo\Documents\ML2 Project\model_artifacts\kmeans_umap.joblib"
CSV_PATH              = r"C:\Users\wisdo\Documents\ML2 Project\model_artifacts\clustered_umap_output.csv"
# ---------------------------

# Load models
kmeans = load(KMEANS_PATH)
umap_model = load(UMAP_MODEL_PATH)

centroids = kmeans.cluster_centers_
print("Centroid shape:", centroids.shape)

# Compute pairwise distances between centroids
def dist(a, b):
    return np.linalg.norm(a - b)

print("\nPairwise distances between cluster centroids:")
print("0–1:", dist(centroids[0], centroids[1]))
print("0–2:", dist(centroids[0], centroids[2]))
print("1–2:", dist(centroids[1], centroids[2]))


