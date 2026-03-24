"""
JobGuard AI — Hugging Face Spaces Gradio App
Fake job posting detector with real ML model inference.
"""
import re
import json
import joblib
import gradio as gr
from pathlib import Path

# Ensure model exists
MODEL_DIR = Path("model_artifacts")
if not (MODEL_DIR / "classifier.pkl").exists():
    print("Model not found. Running train_for_hf.py...")
    import subprocess
    subprocess.run(["python", "train_for_hf.py"], check=True)

# Load model
model = joblib.load(MODEL_DIR / "classifier.pkl")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
scaler = joblib.load(MODEL_DIR / "meta_scaler.pkl")
with open(MODEL_DIR / "model_config.json") as f:
    config = json.load(f)
META_FEATURES = config["meta_features"]
THRESHOLD = config.get("optimal_threshold", 0.47)

# NLTK
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        words = word_tokenize(text)
        stop = set(stopwords.words("english"))
        words = [w for w in words if w not in stop and len(w) > 1]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
    except Exception:
        words = [w for w in text.split() if len(w) > 1]
    return " ".join(words) if words else "empty"


def extract_meta(text):
    t = str(text) if text else ""
    feats = {
        "text_length": len(t),
        "word_count": len(t.split()),
        "has_email": int(bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", t))),
        "has_url": int(bool(re.search(r"https?://", t))),
        "exclamation_count": t.count("!"),
        "caps_ratio": sum(1 for c in t if c.isupper()) / max(len(t), 1),
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
    }
    return [[feats[n] for n in META_FEATURES]]


def predict(job_text: str):
    if not job_text or len(job_text.strip()) < 10:
        return "⚠️ Please enter at least 10 characters.", 0.0, "N/A"
    from scipy.sparse import hstack, csr_matrix
    clean = preprocess_text(job_text)
    X_tfidf = tfidf.transform([clean])
    X_meta = scaler.transform(extract_meta(job_text))
    X = hstack([X_tfidf, csr_matrix(X_meta)])
    prob = float(model.predict_proba(X)[0][1])
    pred = "🚨 FRAUDULENT" if prob >= THRESHOLD else "✅ LEGITIMATE"
    risk = "HIGH" if prob >= 0.80 else "MEDIUM" if prob >= 0.50 else "LOW" if prob >= 0.20 else "VERY LOW"
    return pred, prob, risk


with gr.Blocks(
    title="JobGuard AI — Fake Job Detector",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="footer {display: none !important}"
) as demo:
    gr.Markdown("""
    # 🛡️ JobGuard AI — Fake Job Postings Detector
    
    Paste any job posting below. The model uses **TF-IDF + 9 behavioral signals** (98.4% accuracy, 0.988 ROC-AUC) to detect fraud.
    """)
    with gr.Row():
        inp = gr.Textbox(
            label="Job Title + Description",
            placeholder="Paste the full job posting here... (title, company, description)",
            lines=6,
        )
    with gr.Row():
        btn = gr.Button("🔍 Analyze", variant="primary")
    with gr.Row():
        verdict = gr.Textbox(label="Verdict", interactive=False)
        prob_out = gr.Number(label="Fraud Probability (0–1)", interactive=False)
        risk_out = gr.Textbox(label="Risk Level", interactive=False)
    btn.click(
        fn=lambda t: predict(t) if t else ("Enter text above.", 0.0, "N/A"),
        inputs=inp,
        outputs=[verdict, prob_out, risk_out],
    )
    gr.Markdown("""
    ---
    **Model:** Random Forest (200 trees) · TF-IDF 5K features + 9 meta-features · SMOTE balanced  
    **Dataset:** [Kaggle Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-predicting) · 17,880 postings
    """)

if __name__ == "__main__":
    demo.launch()
