import streamlit as st
import numpy as np
import re
import os
from pathlib import Path
from joblib import load

# ---------- MODEL PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_ARTIFACTS_PATH", BASE_DIR / "model_artifacts"))
EMBED_MODEL_PATH = MODEL_DIR / "embedding_model.joblib"
UMAP_MODEL_PATH = MODEL_DIR / "umap_model.joblib"
KMEANS_PATH = MODEL_DIR / "kmeans_umap.joblib"
# ---------------------------------


# Load models
if not EMBED_MODEL_PATH.exists() or not UMAP_MODEL_PATH.exists() or not KMEANS_PATH.exists():
    st.error("Model files not found. Please check model_artifacts path and ensure joblib files are available.")
    st.stop()

embed_model = load(EMBED_MODEL_PATH)
umap_model = load(UMAP_MODEL_PATH)
kmeans = load(KMEANS_PATH)

# Streamlit page settings
st.set_page_config(
    page_title="Speech Pattern Analyzer",
    layout="wide"
)

st.title("🧠 Unsupervised Speech Pattern Analyzer")
st.write("""
This tool uses **MiniLM Sentence Embeddings → UMAP → K-Means (K=3)**  
to uncover natural speech patterns in a clinically inspired interview dataset.

👉 *This system is NOT diagnostic. It only analyzes linguistic patterns.*
""")


# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text
# -----------------------------------


# ---------- INPUT ----------
st.subheader("Enter speech transcript text:")
user_text = st.text_area(
    "Paste participant-style responses here:",
    height=200,
    placeholder="Example: I am feeling okay today. My day was mostly normal..."
)
# ---------------------------


# ---------- PROCESS ----------
if st.button("Analyze Speech Pattern") and user_text.strip():

    cleaned = clean_text(user_text)

    # Step 1: embedding
    embedding = embed_model.encode([cleaned])

    # Step 2: reduce via UMAP
    umap_vec = umap_model.transform(embedding)

    # Step 3: cluster prediction
    cluster = int(kmeans.predict(umap_vec)[0])

    # Step 4: distances
    centroids = kmeans.cluster_centers_
    distances = [float(np.linalg.norm(umap_vec - c)) for c in centroids]

    st.markdown(f"## 🔍 Predicted Cluster: **{cluster}**")


    # ==========================================================
    #             FINAL, ACCURATE CLUSTER INTERPRETATIONS
    # ==========================================================

    # 🟥 CLUSTER 2 — TYPICAL CONVERSATIONAL SPEECH
    if cluster == 2:
        st.markdown("### 🟥 Cluster 2 — Typical Conversational Speech (Most Common)")
        st.write("""
        This cluster represents the **dominant, everyday conversational style**  
        found in DAIC-like interviews. It captures natural, neutral-flow speech.

        **Characteristics:**
        - Simple, normal conversational statements  
        - Neutral or mildly emotional tone (“okay”, “fine”, “tired”)  
        - Consistent sentence flow  
        - Descriptions of daily routine or general feelings  
        - Stable structure and no large tone shifts  

        This is the **baseline speech pattern**, representing typical dialogue behavior.
        """)


    # 🟦 CLUSTER 0 — REFLECTIVE, STRUCTURED, INSIGHTFUL SPEECH
    elif cluster == 0:
        st.markdown("### 🟦 Cluster 0 — Reflective, Structured, Insight-Oriented Speech")
        st.write("""
        This cluster includes responses with **higher cognitive organization**  
        and **self-reflection**, often showing deeper personal insight.

        **Characteristics:**
        - Reflective (“I’ve been thinking about…”)  
        - Structured, well-organized statements  
        - Personal evaluation or reasoning  
        - Intentional self-analysis  
        - More complex linguistic structure  

        This speech style appears less often but represents **thoughtful, introspective communication**.
        """)


    # 🟩 CLUSTER 1 — MINIMAL, ABSTRACT, DETACHED SPEECH
    elif cluster == 1:
        st.markdown("### 🟩 Cluster 1 — Minimal, Abstract, Detached Speech")
        st.write("""
        This cluster contains **short, vague, or abstract responses**,  
        often lacking elaboration or contextual grounding.

        **Characteristics:**
        - One-word replies (“nothing”, “maybe”, “I don’t know”)  
        - Low elaboration  
        - Ambiguous or detached tone  
        - Conceptual or vague statements  
        - Little to no narrative structure  

        This speech pattern reflects **minimal engagement or abstract thinking**,  
        not a clinical diagnosis.
        """)


    # ---------- DISTANCES ----------
    st.markdown("### 📊 Distances to All Clusters (lower = more similar)")
    st.write({
        "Cluster 0": distances[0],
        "Cluster 1": distances[1],
        "Cluster 2": distances[2]
    })

    # Confidence
    closest_dist = min(distances)
    confidence = 1 / (closest_dist + 1e-6)

    st.markdown(f"### 🔎 Confidence Score: **{confidence:.3f}** (relative similarity)")
