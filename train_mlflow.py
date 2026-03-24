"""
Train JobGuard AI with MLflow experiment tracking.
Run: python train_mlflow.py
"""
import mlflow
import mlflow.sklearn
from pathlib import Path

# Run the training (imports from train_for_hf)
from train_for_hf import main, MODEL_DIR

if __name__ == "__main__":
    mlflow.set_experiment("jobguard-ai")

    with mlflow.start_run():
        main()  # This saves to model_artifacts/

        # Log metrics (re-read from saved config)
        import json
        with open(MODEL_DIR / "model_config.json") as f:
            cfg = json.load(f)
        mlflow.log_metric("accuracy", cfg["accuracy"])
        mlflow.log_metric("roc_auc", cfg["roc_auc"])
        mlflow.log_metric("f1", cfg["f1"])

        # Log artifacts
        mlflow.log_artifact(str(MODEL_DIR / "classifier.pkl"), artifact_path="model")
        mlflow.log_artifact(str(MODEL_DIR / "tfidf_vectorizer.pkl"), artifact_path="model")
        mlflow.log_artifact(str(MODEL_DIR / "meta_scaler.pkl"), artifact_path="model")
        mlflow.log_artifact(str(MODEL_DIR / "model_config.json"), artifact_path="model")

        print("Logged to MLflow.")
