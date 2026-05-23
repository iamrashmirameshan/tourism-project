
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi
import os

repo_id = "rashmipr/tourism-project"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("MY_MLOps_Token"))

# Check if repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")

    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )

    print(f"Dataset repo '{repo_id}' created.")

# Upload dataset folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded successfully.")
