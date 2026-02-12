# Import required libraries
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Initialize the API
api = HfApi(token=os.getenv("HF_TOKEN"))

# repo details on Hugging Face
repo_id = "asvravi/asv-predictive-maintenance-backend"
repo_type = "space"

# Check if the space repo exists
try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating new space...")
  create_repo(repo_id=repo_id, repo_type=repo_type,space_sdk="docker", private=False)
  print(f"Space '{repo_id}' created.")

# Upload Streamlit app files stored in the folder called deployment_files
api.upload_folder(
    folder_path="CS_Preventive_Maintenance/deployment/backend",  #Local folder path
    repo_id=repo_id,  # Hugging face space id
    repo_type="space", # Hugging face repo type "space"
)
