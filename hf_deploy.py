import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))
repo_id = "AishwaryaKa/hr-policy-bot"

print("Creating Hugging Face Space...")
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print("Space ready!")
except Exception as e:
    print(f"Creation response: {e}")

print("Uploading backend folder to Space...")
try:
    api.upload_folder(
        folder_path="./backend",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["venv/*", "__pycache__/*", "*.pyc", ".env"]
    )
    print("Upload completed successfully!")
except Exception as e:
    print(f"Error uploading: {e}")
