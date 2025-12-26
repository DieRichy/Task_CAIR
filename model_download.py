# Function to download a model
import os
from huggingface_hub import snapshot_download
def download_model(model_name, local_dir, ignore_patterns=None, hf_token=None):
    # Ensure the directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Download the model
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Do not use symlinks
        token=hf_token,  # Use the provided Hugging Face token
        ignore_patterns=ignore_patterns,  # Ignore unnecessary files
        force_download=True
    )

    print(f"Model has been downloaded to: {local_dir}")

hf_token = os.environ.get("HF_TOKEN")

download_model(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    local_dir="your path/models/Qwen2.5-3B-Instruct",
    hf_token=hf_token
    )
