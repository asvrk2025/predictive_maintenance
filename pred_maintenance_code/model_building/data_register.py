
# Import required libraries
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Hugging Face repo details
repo_id = "asvravi/asv-preventive-maintenance"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists, if not create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload data folder to Hugging face
api.upload_folder(
    folder_path="pred_maintenance_code/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
