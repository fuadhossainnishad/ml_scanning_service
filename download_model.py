from huggingface_hub import snapshot_download
import os

LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fashion-clip")

print("⏳ Downloading Fashion-CLIP model...")
snapshot_download(
    repo_id="patrickjohncyh/fashion-clip",
    local_dir=LOCAL_DIR
)
print(f"✅ Model downloaded to: {LOCAL_DIR}")