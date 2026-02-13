"""
Preparation script for Engine Predictive Maintenance project.
Responsibilities:
- Environment validation
- Data loading from Hugging Face
- Schema & sanity checks
- Train-test split (stratified)
- Persist processed datasets & config
"""
# Import required libraries
import os
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "engine_condition"

EXPECTED_COLUMNS = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
    "engine_condition",
]

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CONFIG_DIR = "config"

# Create directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# -----------------------------
# Environment checks
# -----------------------------
print("🔍 Checking environment variables...")

required_envs = ["HF_TOKEN"]
missing = [env for env in required_envs if env not in os.environ]

if missing:
    raise RuntimeError(f"Missing required environment variables: {missing}")

print("✅ Environment OK")

# -----------------------------
# Load dataset from Hugging Face
# -----------------------------
print("📥 Loading dataset from Hugging Face...")

# Define constants for the dataset path
DATASET_PATH = "hf://datasets/asvravi/asv-preventive-maintenance/engine_data.csv"

# Reading data from Hugging face
df = pd.read_csv(DATASET_PATH)

df.to_csv(os.path.join(RAW_DIR, "engine_data.csv"), index=False)

print(f"✅ Dataset loaded: {df.shape}")

# -----------------------------
# Schema validation
# -----------------------------
print("🧪 Validating dataset schema...")

# Standardize column names: strip leading/trailing spaces, convert to lowercase,
# and replace one or more non-word characters with a single underscore
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[^\w]+", "_", regex=True)
      .str.strip("_")
)


if set(df.columns) != set(EXPECTED_COLUMNS):
    raise ValueError(
        f"Schema mismatch.\nExpected: {EXPECTED_COLUMNS}\nFound: {list(df.columns)}"
    )

if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains null values")

print("✅ Schema validated")

# -----------------------------
# Train-test split (stratified)
# -----------------------------
print("✂️ Performing stratified train-test split...")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Save processed datasets
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("✅ Train-test split completed")
print(f"X_Train size: {X_train.shape}, X_Test size: {X_test.shape}")
print(f"y_train size: {y_train.shape}, y_test size: {y_test.shape}")

# -----------------------------
# Save configuration
# -----------------------------
print("🧾 Saving configuration...")

config = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "target_column": TARGET_COL,
    "features": list(X.columns),
    "class_distribution": y.value_counts(normalize=True).to_dict(),
}

with open(os.path.join(CONFIG_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("✅ Config saved")

# -----------------------------
# Summary
# -----------------------------
print("\n🎯 prep.py completed successfully")
print("Artifacts generated:")
print("- data/raw/engine_data.csv")
print("- data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("- config/config.json")

# -----------------------------
# Upload to Hugging Face
# -----------------------------
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# uploading the train and test csv files to Hugging Face
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="asvravi/asv-preventive-maintenance",
        repo_type="dataset",
    )
