# Unsupervised Speech Pattern Analysis

A Streamlit application for unsupervised speech transcript analysis using MiniLM embeddings, UMAP dimension reduction, and K-Means clustering.

## Project Structure

- `app.py`: Streamlit interface and prediction pipeline.
- `train_umap_embeddings.py`: training pipeline for embeddings/UMAP/KMeans.
- `exe.py`: internal execution helper (if used).
- `model_artifacts/`: folder containing trained models (`embedding_model.joblib`, `umap_model.joblib`, `kmeans_umap.joblib`).
- `Transcripts/`: sample transcript CSV data.
- `requirements.txt`: Python dependencies.

## Local setup

```powershell
cd "C:\Users\wisdo\Documents\ML2 Project"
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run locally

```powershell
streamlit run app.py
```

### If your models are not in local `model_artifacts/` (recommended for deployment):

- Put model files in `model_artifacts/`, or
- Set environment variable:

```powershell
$env:MODEL_ARTIFACTS_PATH = "C:\path\to\model_artifacts"
streamlit run app.py
```

## GitHub

Repository: https://github.com/Arpita-2755/unsupervised-speech-pattern-analysis

## Deployment

1. Ensure `requirements.txt` is complete.
2. Push to GitHub.
3. Use Streamlit Community Cloud (free):
   - Visit https://streamlit.io/cloud
   - Connect GitHub account
   - Add app: select repo, branch `main`, file `app.py`
4. Alternative free hosts:
   - Render.com, Railway.app, Fly.io, Hugging Face Spaces

## Notes

- This is a prototype and non-diagnostic.
- Avoid storing large model binaries on GitHub; keep in cloud storage and/or use GitHub releases or LFS.
