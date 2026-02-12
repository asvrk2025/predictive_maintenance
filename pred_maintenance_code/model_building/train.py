import os
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# -----------------------------
# Config
# -----------------------------
RANDOM_STATE = 42
MODEL_NAME = "xgb_tuned_2"
MODEL_FILE = "predictive_maintenance_model_v1.joblib"
HF_REPO_ID = "asvravi/asv-predictive-maintenance"
HF_REPO_TYPE = "model"

DATA_DIR = "data/processed"   # adjust if needed
X_TRAIN_PATH = f"{DATA_DIR}/X_train.csv"
X_TEST_PATH  = f"{DATA_DIR}/X_test.csv"
Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"
Y_TEST_PATH  = f"{DATA_DIR}/y_test.csv"


# -----------------------------
# Load Data
# -----------------------------
print("Loading training and test data...")

Xtrain = pd.read_csv(X_TRAIN_PATH)
Xtest  = pd.read_csv(X_TEST_PATH)
ytrain = pd.read_csv(Y_TRAIN_PATH).squeeze()
ytest  = pd.read_csv(Y_TEST_PATH).squeeze()
print("✅ Data loaded successfully")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape :", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape :", ytest.shape)

# -------------------------------------------------
# Preprocessing Note:
# No additional preprocessing (scaling / normalization)
# is applied here because:
# 1. XGBoost is a tree-based model and does not
#    require feature scaling.
# 2. All features are numeric, continuous, and
#    already standardized in naming.
# 3. Using raw sensor units preserves domain
#    interpretability (RPM, pressure, temperature).
# 4. Data was validated during tuning and showed
#    no performance improvement with scaling.
# -------------------------------------------------

# -----------------------------
# Define Final Model (XGBoost Tuned-2)
# -----------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    n_estimators=300,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,
    min_child_weight=3,
    reg_alpha=0.0,
    reg_lambda=1.0,
    scale_pos_weight=1.2
)


# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_experiment("CS_Engine_Predictive_Maintenance")

with mlflow.start_run(run_name=MODEL_NAME):

    # -----------------------------
    # Train Model
    # -----------------------------
    print("Training model...")
    model.fit(Xtrain, ytrain)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(Xtest)
    y_proba = model.predict_proba(Xtest)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "accuracy": accuracy_score(ytest, y_pred),
        "precision": precision_score(ytest, y_pred),
        "recall": recall_score(ytest, y_pred),  # recall for faulty class (1)
        "f1_score": f1_score(ytest, y_pred),
        "roc_auc": roc_auc_score(ytest, y_proba)
    }

    print("\n📊 Test Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n📌 Confusion Matrix")
    print(confusion_matrix(ytest, y_pred))

    print("\n📌 Classification Report")
    print(classification_report(ytest, y_pred))

    # Log metrics
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # -----------------------------
    # Log Parameters
    # -----------------------------
    mlflow.log_params(model.get_params())

    # -----------------------------
    # Log Artifacts
    # -----------------------------
    mlflow.log_text(
        str(confusion_matrix(ytest, y_pred)),
        artifact_file="confusion_matrix.txt"
    )

    mlflow.log_text(
        classification_report(ytest, y_pred),
        artifact_file="classification_report.txt"
    )

    # -----------------------------
    # Save Model Locally
    # -----------------------------
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved locally as {MODEL_FILE}")

    mlflow.log_artifact(MODEL_FILE, artifact_path="model")

    # Log model to MLflow Model Registry
    mlflow.xgboost.log_model(
        model,
        name="xgb_model",
        input_example=Xtrain.iloc[:5],
        registered_model_name="Engine_Predictive_Maintenance_Model"
    )


# -----------------------------
# Upload Model to Hugging Face
# -----------------------------
print("Uploading model to Hugging Face...")

api = HfApi()

try:
    api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    print("HF model repo exists. Using existing repo.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        private=False
    )
    print("HF model repo created.")

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE
)

print("✅ Training, logging, and model upload completed successfully.")
