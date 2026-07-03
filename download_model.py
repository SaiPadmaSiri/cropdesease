import os
import requests

def ensure_model_present(dest_path: str):
    """Ensure the model file exists at dest_path. If missing, download from MODEL_URL env var."""
    if os.path.exists(dest_path):
        return
    url = os.getenv("MODEL_URL")
    if not url:
        raise RuntimeError("Model file not found and MODEL_URL environment variable is not set.")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Basic verification
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        raise RuntimeError("Downloaded model file is empty or missing")
