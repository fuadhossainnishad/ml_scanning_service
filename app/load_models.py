from transformers import CLIPProcessor, CLIPModel
import os


def load_clip_model(LOCAL_MODEL_PATH: str, device):
    """
    Load CLIP model and processor.
    If the local path exists, load from it.
    Otherwise, download from HuggingFace and cache automatically.
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        model = CLIPModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
        processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_PATH)
        print(f"✅ Loaded CLIP model from local path: {LOCAL_MODEL_PATH}")
    else:
        print("⚠️ Local model not found. Downloading from HuggingFace...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ Downloaded CLIP model from HuggingFace")
    return model, processor
