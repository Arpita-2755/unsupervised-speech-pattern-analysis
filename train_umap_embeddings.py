import os
import glob
import re
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
from joblib import dump

# ---------- PATHS ----------
TRANSCRIPTS_DIR = r"C:\Users\wisdo\Documents\ML2 Project\Transcripts"
OUTPUT_DIR = r"C:\Users\wisdo\Documents\ML2 Project\model_artifacts"
# ---------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------- 1. READ TRANSCRIPTS ----------
records = []

for filepath in glob.glob(os.path.join(TRANSCRIPTS_DIR, "*_TRANSCRIPT.*")):
    fname = os.path.basename(filepath)
    try:
        pid = int(fname.split("_")[0])
    except:
        continue

    df = pd.read_csv(filepath, sep="\t")

    if "speaker" not in df.columns or "value" not in df.columns:
        print("Skipping:", fname)
        continue

    participant_rows = df[df["speaker"].str.contains("Participant", case=False, na=False)]
    text = " ".join(str(x) for x in participant_rows["value"].tolist())
    text = clean_text(text)

    if len(text.split()) < 20:
        continue

    records.append({"participant_id": pid, "text": text})

df_text = pd.DataFrame(records)
print("Total participants loaded:", len(df_text))


# ---------- 2. EMBEDDINGS ----------
print("\nLoading MiniLM embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Computing embeddings...")
embeddings = model.encode(df_text["text"].tolist(), show_progress_bar=True)
print("Embedding shape:", embeddings.shape)


# ---------- 3. UMAP DIMENSION REDUCTION ----------
print("\nReducing dimensions using UMAP...")
umap_model = umap.UMAP(
    n_components=10,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

umap_embeddings = umap_model.fit_transform(embeddings)
print("UMAP reduced shape:", umap_embeddings.shape)


# ---------- 4. KMEANS CLUSTERING ----------
K = 3
print(f"\nClustering with K = {K}...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=50)
cluster_labels = kmeans.fit_predict(umap_embeddings)
df_text["cluster"] = cluster_labels

print("\nCluster counts:")
print(df_text["cluster"].value_counts())


# ---------- 5. SAVE EVERYTHING ----------
dump(model, os.path.join(OUTPUT_DIR, "embedding_model.joblib"))
dump(umap_model, os.path.join(OUTPUT_DIR, "umap_model.joblib"))
dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans_umap.joblib"))
df_text.to_csv(os.path.join(OUTPUT_DIR, "clustered_umap_output.csv"), index=False)

print("\nModel saved successfully.")
