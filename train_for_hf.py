"""
Train JobGuard AI model and save artifacts for Hugging Face Spaces.
Run once: python train_for_hf.py
Then commit model_artifacts/ to your repo.
"""
import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# NLTK
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

MODEL_DIR = Path("model_artifacts")
MODEL_DIR.mkdir(exist_ok=True)

META_FEATURES = [
    "text_length", "word_count", "has_email", "has_url", "exclamation_count",
    "caps_ratio", "telecommuting", "has_company_logo", "has_questions"
]


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


def main():
    # Load data
    paths = ["jobguard-dataset.csv", "data/jobguard-dataset.csv"]
    df = None
    for p in paths:
        if Path(p).exists():
            df = pd.read_csv(p)
            print(f"Loaded: {p} ({len(df):,} rows)")
            break
    if df is None:
        raise FileNotFoundError("jobguard-dataset.csv not found. Place it in repo root.")

    # Text data
    df["text_data"] = (
        df["title"].fillna("") + " " +
        df["company_profile"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["requirements"].fillna("") + " " +
        df["benefits"].fillna("")
    ).str.strip().str.replace(r"\s+", " ", regex=True)

    # Meta features
    t = df["text_data"]
    df["text_length"] = t.str.len()
    df["word_count"] = t.str.split().str.len().fillna(0).astype(int)
    df["has_email"] = t.str.contains(r"[\w.+-]+@[\w-]+\.[\w.-]+", regex=True, na=False).astype(int)
    df["has_url"] = t.str.contains("http", case=False, na=False).astype(int)
    df["exclamation_count"] = t.str.count(r"!").fillna(0).astype(int)
    df["caps_ratio"] = t.apply(lambda s: sum(1 for c in str(s) if c.isupper()) / max(len(str(s)), 1))
    df["telecommuting"] = df["telecommuting"].fillna(0).astype(int)
    df["has_company_logo"] = df["has_company_logo"].fillna(0).astype(int)
    df["has_questions"] = df["has_questions"].fillna(0).astype(int)

    # Preprocess
    df["clean_text"] = df["text_data"].apply(preprocess_text)
    df["clean_text"] = df["clean_text"].replace("empty", "unknown job posting")

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
        min_df=3, max_df=0.90, strip_accents="unicode", analyzer="word"
    )
    X_tfidf = tfidf.fit_transform(df["clean_text"])

    # Meta
    scaler = MaxAbsScaler()
    X_meta = scaler.fit_transform(df[META_FEATURES].fillna(0))
    X = hstack([X_tfidf, csr_matrix(X_meta)])
    y = df["fraudulent"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    if SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Eval
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

    # Save
    joblib.dump(clf, MODEL_DIR / "classifier.pkl")
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(scaler, MODEL_DIR / "meta_scaler.pkl")
    config = {
        "optimal_threshold": 0.47,
        "meta_features": META_FEATURES,
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "f1": float(f1),
    }
    with open(MODEL_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
