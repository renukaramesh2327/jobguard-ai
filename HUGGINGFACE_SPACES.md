# Deploy JobGuard AI on Hugging Face Spaces

## Step 1: Train the model (one-time)

```bash
python train_for_hf.py
```

This creates `model_artifacts/` with the trained classifier.

## Step 2: Create your Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Name: `jobguard-ai` (or similar)
4. SDK: **Gradio**
5. Hardware: **CPU basic** (free)

## Step 3: Add files

Upload or push these files to your Space:

- `app.py` — Gradio interface
- `train_for_hf.py` — Training script (runs automatically if model missing)
- `requirements.txt` — Dependencies (includes gradio)
- `jobguard-dataset.csv` — Training data
- `model_artifacts/` — Trained model (classifier.pkl, tfidf_vectorizer.pkl, meta_scaler.pkl, model_config.json)

## Step 4: Deploy

HF Spaces auto-builds. Your app will be live at:

`https://YOUR_USERNAME-jobguard-ai.hf.space`

Add this URL to your portfolio so recruiters can try the real model.
